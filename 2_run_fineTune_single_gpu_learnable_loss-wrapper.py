# Single-GPU version with learnable task weights
# Simplified version without DDP complexity

import os
import logging
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import time
import json
import math

# Import utilities from local model_utils_single_gpu.py file
from model_utils_single_gpu import (
    DinoVisionTransformer,
    VolumeProcessor,
    NLSTDataset,
    calculate_class_weights,
    install_packages,
    prepare_balanced_validation,
    calculate_auc,
    MetricsLogger,
    MetricsCalculator,
    augment_3_channel,
    ModelSaver,
    load_latest_checkpoint,
    make_balanced_sampler,
    print_training_dataset_stats,
    CombinedModel_weight,
    UncertaintyWeightedSumNamed,
    _confusion_metrics_from_cm,
    _exactly_one_hot_valid,
    _print_group_table,
    focal_ce
)

def parse_groups_list(groups_str):
    """Parse groups from command line format: '[[a,b,c], [d], [e,f,g]]'"""
    import ast
    try:
        # Parse the string as a Python literal (list of lists)
        groups = ast.literal_eval(groups_str)
        if not isinstance(groups, list):
            raise ValueError("Input must be a list")
        for group in groups:
            if not isinstance(group, list):
                raise ValueError("Each group must be a list")
        return groups
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid groups format: {groups_str}. Expected format: '[[a,b,c], [d], [e,f,g]]'. Error: {e}")

def parse_groups(groups_list):
    """Parse groups from command line format: ['group1,group2', 'group3', 'group4,group5,group6']"""
    parsed_groups = []
    for group_str in groups_list:
        # Split by comma and strip whitespace
        group_labels = [label.strip() for label in group_str.split(',')]
        parsed_groups.append(group_labels)
    return parsed_groups

def build_index_groups(label_cols, groups_def, binary_names):
    """Map names->indices for groups + binaries, based on your label_cols order."""
    name2idx = {n: i for i, n in enumerate(label_cols)}
    idx_groups = {}
    for g, names in groups_def.items():
        idxs = [name2idx[n] for n in names if n in name2idx]
        if len(idxs) != len(names):
            missing = set(names) - set(name2idx)
            raise ValueError(f"[{g}] missing in label_cols: {missing}")
        idx_groups[g] = idxs
    binary_idxs = [name2idx[n] for n in binary_names if n in name2idx]
    return idx_groups, binary_idxs

def unfreeze_after_heads(model):
    """
    Freeze all params, then unfreeze:
      - last block's attention output projection (attn.proj)
      - last block's ls1, norm2, mlp, ls2
      - model.norm (final LN)
      - model.head
    """

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Locate the very last transformer block
    last_block = model.blocks[-1][-1]

    # 1) Unfreeze the attention output projection
    for p in last_block.attn.proj.parameters():
        p.requires_grad = True

    # 2) Unfreeze modules after attention
    for m in [last_block.ls1, last_block.norm2, last_block.mlp, last_block.ls2,
              model.norm, model.head]:
        for p in m.parameters():
            p.requires_grad = True

    # --- Count and report ---
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", f"{total_trainable:,}")

    print("\nTrainable parameter breakdown:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:60s} {p.numel():,}")

    return total_trainable

# Simplified Trainer class for single GPU
class SingleGPUTrainer:
    def __init__(self, args, label_cols, groups_def=None, binary_names=None):
        self.args = args
        self.label_cols = label_cols
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Label smoothing factor for CE on group heads
        self.ce_smoothing = 0.05

        # Use provided group definitions or defaults
        if groups_def is None:
            groups_def = {
                'histology':   ['is_scc', 'is_adenocarcinoma', 'is_other'],
                'metastasis':  ['is_M0', 'is_M1A', 'is_M1B', 'is_M1C'],
                'node':        ['is_N0', 'is_N1', 'is_N2', 'is_N3'],
                'tumor':       ['is_T0', 'is_T1', 'is_T2', 'is_T3', 'is_T4'],
            }
        if binary_names is None:
            binary_names = ['is_mw']

        # Build index groups once
        self.idx_groups, self.binary_idxs = build_index_groups(self.label_cols, groups_def, binary_names)
        self.group_names  = list(self.idx_groups.keys())
        self.binary_names = [self.label_cols[j] for j in self.binary_idxs]
        logging.info(f"Groups: { {g: self.idx_groups[g] for g in self.group_names} }")
        logging.info(f"Binaries: { {n: j for n, j in zip(self.binary_names, self.binary_idxs)} }")

        # Initialize model and training components
        self._init_model()
        self._init_criterion()
        self._init_optimizer()
        self._init_scaler()

        # Misc
        self.eps = 1e-8

        # Initialize metrics components
        self.metrics_logger = MetricsLogger(0, self.args.metrics_dir)
        self.metrics_calculator = MetricsCalculator()

        # Initialize model saver
        self.model_saver = ModelSaver(self.args.output, 0)

        # Initialize running accuracy metrics
        self.tp_preds = {}
        self.fn_preds = {}
        self.tn_preds = {}
        self.fp_preds = {}
        self.reset_running_metrics()

        # Logging/validation guards based on optimizer steps
        self.last_logged_step = -1
        self.last_validated_step = -1
        self._prev_global_step = -1

        # Print configuration
        print("\nTraining Configuration:")
        print(f"split_col: {self.args.split_col}")
        print(f"Max chunks per sample: {self.args.max_chunks}")
        print(f"Learning rate: {self.args.lr}")
        print(f"Gradient accumulation steps: {self.args.accum_steps}")
        print(f"Number of tasks: {len(self.label_cols)}")
        print(f"Validation frequency: {self.args.val_every} steps")
        print(f"Number of epochs: {self.args.epochs}")
        print(f"Warmup steps: {self.args.warmup_steps}")
        print(f"LSE tau (temperature): {self.args.lse_tau}")
        print(f"Waqas way: {'Enabled' if self.args.waqas_way else 'Disabled'}")
        print(f"Device: {self.device}")
        print(f"Model initialization: {'Pretrained weights' if self.used_pretrained_weights else 'Random initialization'}")
        print(f"Balanced sampling: {'Enabled' if self.args.balanced_sampling else 'Disabled'}")
        if self.args.no_learnable_weights:
            print(f"Learnable task weights: Disabled (using equal weights)")
        else:
            print(f"Learnable task weights: UncertaintyWeightedSumNamed")
        print()
        logging.info("------ Training Configuration ------")
        logging.info(f"split_col                     : {self.args.split_col}")
        logging.info(f"Max chunks per sample         : {self.args.max_chunks}")
        logging.info(f"Learning rate (aggregator)    : {self.args.lr}")
        logging.info(f"Learning rate (base)          : {self.args.lr*0.2}")
        logging.info(f"Gradient accumulation steps   : {self.args.accum_steps}")
        logging.info(f"Warm-up steps                 : {self.args.warmup_steps}")
        logging.info(f"Aggregator layers / heads     : {self.args.num_layers} / {self.args.num_attn_heads}")
        logging.info(f"Aggregator dropout            : {self.args.dropout}")
        logging.info(f"LSE tau (temperature)         : {self.args.lse_tau}")
        logging.info(f"Waqas way                     : {'Enabled' if self.args.waqas_way else 'Disabled'}")
        logging.info(f"Validation frequency          : {self.args.val_every} steps")
        logging.info(f"Number of epochs              : {self.args.epochs}")
        logging.info(f"Device                        : {self.device}")
        logging.info(f"Model initialization          : {'Pretrained weights' if self.used_pretrained_weights else 'Random initialization'}")
        logging.info(f"Balanced sampling             : {'Enabled' if self.args.balanced_sampling else 'Disabled'}")
        if self.args.no_learnable_weights:
            logging.info(f"Learnable task weights        : Disabled (using equal weights)")
        else:
            logging.info(f"Learnable task weights        : UncertaintyWeightedSumNamed")
        logging.info(f"Number of tasks               : {len(self.label_cols)}")
        logging.info(f"Task names                    : {self.label_cols}")
        if self.pos_weight is not None:
            logging.info(f"Class weights enabled         : Yes")
            for i, task_name in enumerate(self.label_cols):
                weight_val = self.pos_weight[i].item()
                logging.info(f"  {task_name} pos_weight      : {weight_val:.4f}")
        else:
            logging.info(f"Class weights enabled         : No (unweighted)")
        logging.info("------------------------------------\n")

    def _init_model(self):
        """Initialize the model"""
        sys.path.insert(0, "/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Checkpoints/")

        from functools import partial
        from dinov2.models.vision_transformer import vit_base, DinoVisionTransformer
        from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

        # Load the pretrained model
        checkpoint_path = "/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Checkpoints/training_1124999/teacher_checkpoint.pth"
        patch_size = 16
        
        # Load the base model first on CPU
        model_ct = DinoVisionTransformer(
            img_size=448,
            patch_size=patch_size,
            drop_path_rate=0.0,
            block_chunks=1,
            drop_path_uniform=True,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            block_fn=partial(Block, attn_class=MemEffAttention),
            num_register_tokens=5,
            init_values=1.0e-05,
        )

        # Add explicit initialization of cls_token
        model_ct.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
        torch.nn.init.normal_(model_ct.cls_token, std=0.02)

        # Load the weights - OPTIONAL BASED ON COMMAND LINE ARGUMENT
        print(f"DEBUG: args.no_load_pretrained = {self.args.no_load_pretrained}")

        if not self.args.no_load_pretrained:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            teacher_weights = checkpoint["teacher"]
            teacher_weights_cleaned = {k.replace("backbone.", ""): v for k, v in teacher_weights.items()}
            model_ct.load_state_dict(teacher_weights_cleaned, strict=False)
            print(f"Loaded pretrained weights from: {checkpoint_path}")
            self.used_pretrained_weights = True
        else:
            print(f"Using random initialization (no pretrained weights loaded)")
            self.used_pretrained_weights = False

        # Materialize tokens
        def materialize_tokens_(m, dev):
            with torch.no_grad():
                for n in ["cls_token", "register_tokens", "mask_token"]:
                    if hasattr(m, n) and getattr(m, n).untyped_storage().size() == 0:
                        real = torch.zeros_like(getattr(m, n), device=dev)
                        torch.nn.init.normal_(real, std=0.02)
                        setattr(m, n, torch.nn.Parameter(real, requires_grad=True))

        materialize_tokens_(model_ct, torch.device('cpu'))

        # Build the combined model
        self.model = CombinedModel_weight(
            base_model=model_ct,
            chunk_feat_dim=768,
            hidden_dim=1024,
            num_tasks=len(self.label_cols),
            num_attn_heads=self.args.num_attn_heads,
            num_layers=self.args.num_layers,
            dropout_rate=self.args.dropout,
            lse_tau=self.args.lse_tau,
            waqas_way=self.args.waqas_way
        )

        # === Learnable per-task weights OUTSIDE the model ===
        num_tasks = len(self.label_cols)
        if not self.args.no_learnable_weights:
            # Use group-level names + binary names for the uncertainty wrapper
            task_names   = self.group_names + self.binary_names
            use_half_map = {name: False for name in task_names}  # CE/BCE → no 0.5 factor

            self.loss_wrapper = UncertaintyWeightedSumNamed(
                task_names=task_names,
                use_half_map=use_half_map,
                clamp_logvar=(-10.0, 10.0)
            ).to(self.device)
            self.task_weight_logits = None
            # Store task names for logging
            self.loss_task_names = task_names
            print("UncertaintyWeightedSumNamed: ENABLED")
            print(f"Task names: {task_names}")
            print(f"Use half map: {use_half_map}")
            logging.info("UncertaintyWeightedSumNamed: ENABLED")
            logging.info(f"Task names: {task_names}")
            logging.info(f"Use half map: {use_half_map}")
        else:
            self.task_weight_logits = None
            self.loss_wrapper = None
            self.loss_task_names = []
            print("Learnable task weights: DISABLED (using equal weights)")
            logging.info("Learnable task weights: DISABLED (using equal weights)")

        # Move model to device
        self.model = self.model.to(self.device)

        # Set all parameters to requires_grad=True
        # for param in self.model.parameters():
        #     param.requires_grad = True

        # Apply unfreeze_after_heads to model_ct (base model)
        if not self.args.no_freeze_backbone:
            print("Applying unfreeze_after_heads to DINO backbone...")
            total_trainable = unfreeze_after_heads(model_ct)
            print(f"DINO backbone configured with {total_trainable:,} trainable parameters!")
        else:
            print("DINO backbone will be trainable (all parameters)")

        # Sanity check: prove that base backbone freezing took effect through self.model
        backbone_params = sum(p.numel() for n,p in self.model.named_parameters() if n.startswith("base.") and p.requires_grad)
        agg_params      = sum(p.numel() for n,p in self.model.named_parameters() if not n.startswith("base.") and p.requires_grad)
        print(f"[CHECK] Trainable in backbone: {backbone_params:,}")
        print(f"[CHECK] Trainable in aggregator/head: {agg_params:,}")

        self.start_epoch = load_latest_checkpoint(
            self.model,
            self.args.output,
            device=self.device,
            rank=0
        )

    def _init_criterion(self):
        self.ce_weights = {}  # per-group class weights for CE

        # --- Load train_df only if needed for weights ---
        train_df = None
        if self.args.class_weights == "True":
            train_df = pd.read_csv(self.args.csv)
            train_df = train_df[train_df[self.args.split_col] == "train"]
            # Existing pos_weight vector (for binaries only)
            w = calculate_class_weights(train_df, self.label_cols).to(self.device)
            self.pos_weight = w
        else:
            self.pos_weight = None

        # --- Build CE weights for each softmax group (only when class weights enabled) ---
        if self.args.class_weights == "True":
            eps = 1e-6
            for g, idxs in self.idx_groups.items():
                cls_names = [self.label_cols[i] for i in idxs]
                sub = train_df[cls_names].replace(-1, np.nan)  # ignore missing
                counts = []
                for cname in cls_names:
                    counts.append(np.nansum(sub[cname].values == 1))
                counts = np.array(counts, dtype=np.float32) + eps
                inv = 1.0 / counts
                weights = (inv / inv.sum()) * len(counts)  # normalized inverse freq
                self.ce_weights[g] = torch.tensor(weights, device=self.device, dtype=torch.float32)
        else:
            # No class weighting: tell CE to use uniform weights
            for g in self.idx_groups.keys():
                self.ce_weights[g] = None

        # Criterion handle for BCE on binaries (we compute manually anyway)
        self.crit = torch.nn.BCEWithLogitsLoss(reduction='none')

        if self.pos_weight is not None:
            print(f"Per-class pos_weight: {self.pos_weight.cpu().numpy()}")
            logging.info(f"Per-class pos_weight: {self.pos_weight.cpu().numpy()}")
            for i, task_name in enumerate(self.label_cols):
                weight_val = self.pos_weight[i].item()
                print(f"  {task_name}: pos_weight = {weight_val:.4f}")
                logging.info(f"  {task_name}: pos_weight = {weight_val:.4f}")
        else:
            print("Using unweighted BCEWithLogitsLoss for binaries; CE groups use uniform weights")
            logging.info("Using unweighted BCEWithLogitsLoss for binaries; CE groups use uniform weights")

    def _init_optimizer(self):
        """Initialize the optimizer with parameter groups"""
        # Define learning rates
        main_lr = self.args.lr
        base_lr = self.args.lr * 0.2  # lower LR for pretrained DinoVisionTransformer

        # Separate base and aggregator parameters
        base_params, agg_params = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                (base_params if 'base' in n else agg_params).append(p)

        # include trainer-level task weights (if enabled)
        if self.task_weight_logits is not None:
            agg_params.append(self.task_weight_logits)
        
        # include loss wrapper parameters (if enabled)
        if self.loss_wrapper is not None:
            loss_wrapper_params = list(self.loss_wrapper.parameters())
            agg_params.extend(loss_wrapper_params)
            print(f"Loss wrapper parameters added to optimizer: {len(loss_wrapper_params)} parameters")
            logging.info(f"Loss wrapper parameters added to optimizer: {len(loss_wrapper_params)} parameters")
            # Log the initial log variance values
            if hasattr(self.loss_wrapper, 'log_vars'):
                log_vars = self.loss_wrapper.log_vars.detach().cpu().numpy()
                print(f"Initial log variance values: {log_vars}")
                logging.info(f"Initial log variance values: {log_vars}")
                for i, task_name in enumerate(self.loss_task_names):
                    print(f"  {task_name}: log_var = {log_vars[i]:.4f}")
                    logging.info(f"  {task_name}: log_var = {log_vars[i]:.4f}")

        # Apply weight decay per group
        param_groups = [
            {'params': base_params, 'lr': base_lr, 'weight_decay': self.args.weight_decay},
            {'params': agg_params,  'lr': main_lr,  'weight_decay': self.args.weight_decay},
        ]

        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(param_groups)
        elif self.args.optimizer == 'adamw':
            # weight_decay is specified per-group above
            self.optimizer = torch.optim.AdamW(param_groups)
        else:
            self.optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    def _init_scaler(self):
        """Initialize the gradient scaler for mixed precision training"""
        self.scaler = torch.cuda.amp.GradScaler()

    def _train_step(self, chunks, labels, mask, spacing):
        chunks = chunks.squeeze(1).to(self.device)
        target = torch.tensor(labels, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)

        if chunks.size(0) > self.args.max_chunks:
            mid = chunks.size(0) // 2
            start = max(0, mid - self.args.max_chunks // 2)
            end   = min(chunks.size(0), start + self.args.max_chunks)
            chunks = chunks[start:end]

        # 1) forward under autocast
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = self.model(chunks, spacing)          # [1, T]

        # 2) FP32 path for weights & loss
        logits = logits.float()
        T = logits.size(1)
        
        # ----- Build loss_dict with CE for groups + BCE for independent binaries -----
        loss_dict = {}

        # 5a) Softmax groups → Cross Entropy on grouped logits
        with torch.no_grad():
            target_t = target.clone()

        for g, idxs in self.idx_groups.items():
            # Skip group if any label in group is missing
            if any((target_t[i].item() == -1) for i in idxs):
                continue

            # Expect one-hot across the group's columns (sum==1)
            # If multiple 1s (noisy), argmax will pick one.
            tgt_slice = target_t[idxs]        # [C]
            tgt_class = torch.argmax(tgt_slice).long()  # scalar

            lg = logits[0, idxs]              # [C]
            # Optionally mask out invalid classes if you have label masks; not needed here.

            ce_w = self.ce_weights.get(g, None)
            if getattr(self.args, 'use_focal_loss', False):
                ce_loss = focal_ce(lg.unsqueeze(0), tgt_class.unsqueeze(0), alpha=ce_w, gamma=2.0)
            else:
                ce_loss = F.cross_entropy(
                    lg.unsqueeze(0), tgt_class.unsqueeze(0),
                    weight=ce_w, label_smoothing=self.ce_smoothing
                )
            loss_dict[g] = ce_loss

        # 5b) Independent binaries → BCE
        for j, name in zip(self.binary_idxs, self.binary_names):
            if mask_tensor[j] and target_t[j] != -1:
                pw = None
                if self.pos_weight is not None:
                    pw = self.pos_weight[j:j+1].float().to(self.device)
                bce = F.binary_cross_entropy_with_logits(
                    logits[0, j:j+1], target_t[j:j+1], pos_weight=pw, reduction='mean'
                )
                loss_dict[name] = bce

        # 5c) Combine via uncertainty wrapper (skips missing keys automatically)
        if self.loss_wrapper is not None:
            total_loss = self.loss_wrapper(loss_dict)
            
            # Log loss wrapper details periodically (avoid spamming at global_step==0 during warmup/accum)
            if self.global_step > 0 and self.global_step % (self.args.print_every * 5) == 0:  # Every 5 print intervals
                with torch.no_grad():
                    if hasattr(self.loss_wrapper, 'log_vars'):
                        log_vars = self.loss_wrapper.log_vars.detach().cpu().numpy()
                        precisions = torch.exp(-self.loss_wrapper.log_vars).detach().cpu().numpy()
                        print(f"Loss wrapper status at step {self.global_step}:")
                        logging.info(f"Loss wrapper status at step {self.global_step}:")
                        for i, task_name in enumerate(self.loss_task_names):
                            shown = loss_dict.get(task_name, None)
                            loss_val = (shown.item() if isinstance(shown, torch.Tensor) else float('nan'))
                            print(f"  {task_name}: log_var={log_vars[i]:.4f}, precision={precisions[i]:.4f}, loss={loss_val}")
                            logging.info(f"  {task_name}: log_var={log_vars[i]:.4f}, precision={precisions[i]:.4f}, loss={loss_val}")
        else:
            # Use equal weights (when learnable weights are disabled)
            total_loss = torch.stack(list(loss_dict.values())).mean()
        total_loss = total_loss / float(self.args.accum_steps)     # grad-accum scaling

        # 3) backward pass
        # Check if we have any active losses
        has_active_losses = len(loss_dict) > 0
        if has_active_losses:
            self.scaler.scale(total_loss).backward()

        # ----- Running metrics -----
        with torch.no_grad():
            # Groups: accuracy via argmax
            for g, idxs in self.idx_groups.items():
                # skip if any label missing
                if any((target[i].item() == -1) for i in idxs):
                    continue
                pred_class = torch.argmax(logits[0, idxs]).item()
                true_class = torch.argmax(target[idxs]).item()
                key = f'grp_{g}'
                if pred_class == true_class:
                    self.correct_preds.setdefault(key, 0); self.correct_preds[key] += 1
                self.total_preds.setdefault(key, 0); self.total_preds[key] += 1

            # Binaries: keep your TP/TN/Sens/Spec logic
            for j, name in zip(self.binary_idxs, self.binary_names):
                if mask_tensor[j] and target[j] != -1:
                    prob = torch.sigmoid(logits[0, j]).item()
                    pred = 1.0 if prob > 0.5 else 0.0
                    true = float(target[j].item())

                    tkey = f'bin_{name}'
                    if pred == true:
                        self.correct_preds.setdefault(tkey, 0); self.correct_preds[tkey] += 1
                    self.total_preds.setdefault(tkey, 0); self.total_preds[tkey] += 1

                    # confusion tallies
                    self.tp_preds.setdefault(tkey, 0); self.fn_preds.setdefault(tkey, 0)
                    self.tn_preds.setdefault(tkey, 0); self.fp_preds.setdefault(tkey, 0)
                    if pred == 1.0 and true == 1.0: self.tp_preds[tkey] += 1
                    elif pred == 0.0 and true == 1.0: self.fn_preds[tkey] += 1
                    elif pred == 0.0 and true == 0.0: self.tn_preds[tkey] += 1
                    elif pred == 1.0 and true == 0.0: self.fp_preds[tkey] += 1

        # Emit running stats in the dict (group acc + binary acc/sens/spec)
        running_accuracies = {}
        for g in self.group_names:
            key = f'grp_{g}'
            if self.total_preds.get(key, 0) > 0:
                running_accuracies[f'acc_{g}_running'] = self.correct_preds[key] / self.total_preds[key]
                # For groups, calculate macro-averaged sensitivity and specificity
                # This requires per-class TP/TN/FP/FN which we don't track in training
                # We'll set to NaN for now - could be implemented with per-class confusion matrix
                running_accuracies[f'sens_{g}_running'] = float('nan')
                running_accuracies[f'spec_{g}_running'] = float('nan')

        for name in self.binary_names:
            tkey = f'bin_{name}'
            if self.total_preds.get(tkey, 0) > 0:
                running_accuracies[f'acc_{name}_running'] = self.correct_preds[tkey] / self.total_preds[tkey]
                tp, fn = self.tp_preds[tkey], self.fn_preds[tkey]
                tn, fp = self.tn_preds[tkey], self.fp_preds[tkey]
                sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
                running_accuracies[f'sens_{name}_running'] = sens
                running_accuracies[f'spec_{name}_running'] = spec

        return {
            'loss': float(total_loss.item()) if has_active_losses else float('nan'),
            'lr': self.scheduler.get_last_lr()[1],
            'spacing': spacing,
            **running_accuracies
        }

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        samples_evaluated = 0

        # ---- GROUP accumulators: KxK CMs + (optional) per-class OvR AUC data ----
        group_cm = {}              # g -> KxK int matrix
        group_class_names = {}     # g -> [label names in order]
        group_auc_scores = {}      # g -> list of lists: scores[k] = list of probs for class k
        group_auc_targets = {}     # g -> list of lists: targets[k] = list of (0/1) ovR truth for class k

        for g, idxs in self.idx_groups.items():
            K = len(idxs)
            group_cm[g] = np.zeros((K, K), dtype=np.int64)
            group_class_names[g] = [self.label_cols[i] for i in idxs]
            group_auc_scores[g]  = [list() for _ in range(K)]
            group_auc_targets[g] = [list() for _ in range(K)]

        # For binaries, reuse arrays for AUC computation
        bin_scores  = {n: [] for n in self.binary_names}
        bin_targets = {n: [] for n in self.binary_names}

        for chunks, labels, mask, spacing in val_loader:
            chunks = chunks.squeeze(1).to(self.device)

            # Apply max_chunks constraint
            if chunks.size(0) > self.args.max_chunks:
                mid_idx = chunks.size(0) // 2
                start_idx = max(0, mid_idx - self.args.max_chunks // 2)
                end_idx = min(chunks.size(0), start_idx + self.args.max_chunks)
                chunks = chunks[start_idx:end_idx]

            logits = self.model(chunks, spacing)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            target = torch.tensor(labels, dtype=torch.float32, device=self.device)

            # ---- per group: update CM and AUC buffers with a strict validity check ----
            with torch.no_grad():
                for g, idxs in self.idx_groups.items():
                    tgt_slice = target[idxs].detach().cpu().numpy()
                    if not _exactly_one_hot_valid(tgt_slice):
                        continue

                    true_class = int(np.argmax(tgt_slice))
                    group_logits = logits[0, idxs]
                    pred_class = int(torch.argmax(group_logits).item())

                    # confusion matrix
                    group_cm[g][true_class, pred_class] += 1

                    # per-class OvR AUC buffers (softmax over the group's logits)
                    probs = torch.softmax(group_logits, dim=0).detach().cpu().numpy()
                    for k in range(len(idxs)):
                        group_auc_scores[g][k].append(float(probs[k]))
                        group_auc_targets[g][k].append(1 if k == true_class else 0)

            # ---- binaries as you already had ----
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
            for j, name in zip(self.binary_idxs, self.binary_names):
                if mask_tensor[j] and target[j] != -1:
                    bin_scores[name].append(torch.sigmoid(logits[0, j]).item())
                    bin_targets[name].append(int(target[j].item()))

            samples_evaluated += 1

        # ---- Build metrics dict for groups from their CMs ----
        metrics = {'samples_evaluated': samples_evaluated, 'avg_loss': 0.0}
        for g, cm in group_cm.items():
            # accuracy from CM
            total = cm.sum()
            acc = (np.trace(cm) / total) if total > 0 else np.nan
            metrics[f'acc_{g}'] = acc

            stats = _confusion_metrics_from_cm(cm)
            class_names = group_class_names[g]

            # print a neat per-class table
            _print_group_table(g, class_names, stats)

            # optional: stash CM (as list for JSON) + per-class metrics into metrics dict
            metrics[f'cm_{g}'] = cm.tolist()

            for i, cname in enumerate(class_names):
                def as_float(x): 
                    try:
                        return float(x) if np.isfinite(x) else float('nan')
                    except Exception:
                        return float('nan')

                metrics[f'support_{g}_{cname}'] = int(stats['support'][i])
                metrics[f'sens_{g}_{cname}']    = as_float(stats['sens'][i])
                metrics[f'spec_{g}_{cname}']    = as_float(stats['spec'][i])
                metrics[f'prec_{g}_{cname}']    = as_float(stats['prec'][i])
                metrics[f'f1_{g}_{cname}']      = as_float(stats['f1'][i])
                metrics[f'balacc_{g}_{cname}']  = as_float(stats['bal_acc'][i])

            # macro/weighted summaries
            for key in ["sens", "spec", "prec", "f1", "bal_acc"]:
                metrics[f'macro_{key}_{g}']    = float(stats['macro'][key]) if np.isfinite(stats['macro'][key]) else float('nan')
                metrics[f'weighted_{key}_{g}'] = float(stats['weighted'][key]) if np.isfinite(stats['weighted'][key]) else float('nan')

            # (Optional) per-class OvR AUC using softmax probs
            for i, cname in enumerate(class_names):
                y_true = np.array(group_auc_targets[g][i], dtype=int)
                y_score = np.array(group_auc_scores[g][i], dtype=float)
                if y_true.size > 0 and len(np.unique(y_true)) == 2:
                    try:
                        auc = roc_auc_score(y_true, y_score)
                    except Exception:
                        auc = np.nan
                else:
                    auc = np.nan
                metrics[f'auc_{g}_{cname}'] = float(auc) if np.isfinite(auc) else float('nan')

            # Pretty print CM
            print(f"\n[{g}] confusion matrix (rows=true, cols=pred):")
            header = " " * 12 + " ".join([f"{c[:8]:>9s}" for c in class_names])
            print(header)
            for i, cname in enumerate(class_names):
                row = " ".join([f"{v:9d}" for v in cm[i]])
                print(f"{cname[:10]:>10s}  {row}")
            
            # Log confusion matrix
            logging.info(f"[{g}] confusion matrix (rows=true, cols=pred):")
            logging.info(header)
            for i, cname in enumerate(class_names):
                row = " ".join([f"{v:9d}" for v in cm[i]])
                logging.info(f"{cname[:10]:>10s}  {row}")

        # Binaries → AUC/ACC/Sens/Spec
        for name in self.binary_names:
            y_true = np.array(bin_targets[name], dtype=np.int32)
            y_score = np.array(bin_scores[name], dtype=np.float32)
            if y_true.size == 0 or len(np.unique(y_true)) < 2:
                metrics[f'auc_{name}']  = np.nan
                metrics[f'acc_{name}']  = np.nan
                metrics[f'sens_{name}'] = np.nan
                metrics[f'spec_{name}'] = np.nan
                continue
            auc = roc_auc_score(y_true, y_score)
            y_pred = (y_score > 0.5).astype(int)
            acc = (y_pred == y_true).mean()
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            metrics[f'auc_{name}']  = auc
            metrics[f'acc_{name}']  = acc
            metrics[f'sens_{name}'] = sens
            metrics[f'spec_{name}'] = spec

        print(f"Evaluated {samples_evaluated} samples")

        self.model.train()
        return metrics

    def fit(self, train_loader, val_loader, epochs):
        """Train the model"""
        self.global_step = 0
        self.current_epoch = 0

        optim_steps_per_epoch = math.ceil(len(train_loader) / self.args.accum_steps)
        total_optim_steps     = optim_steps_per_epoch * epochs
        if self.args.warmup_steps > total_optim_steps:
            raise ValueError("warmup_steps exceeds total optimiser steps.")

        pct_start = max(1e-4, min(0.3, self.args.warmup_steps / max(1, total_optim_steps)))

        # Previous OneCycleLR (kept for reference):
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=[self.args.lr * 0.5, self.args.lr],  # base-model, aggregator
        #     total_steps=total_optim_steps,
        #     pct_start=pct_start,
        #     div_factor=10.0,
        #     final_div_factor=100.0,
        # )

        # Cosine annealing with warm restarts (gentler than OneCycle)
        t0 = max(50, total_optim_steps // 4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=t0,
            T_mult=1,
            eta_min=self.args.lr * 0.01,
        )

        # Record initial LR per param group for correct warmup scaling
        for pg in self.optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

        # Initialize timers
        self._last_batch_end = time.time()
        self.fetch_avg   = 0.0
        self.compute_avg = 0.0
        beta = 0.98

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.metrics_calculator.reset()

            # Reset running metrics at the start of each epoch
            self.reset_running_metrics()

            # Ensure we only log/validate once per optimizer step
            if not hasattr(self, 'last_logged_step'):
                self.last_logged_step = -1
            if not hasattr(self, 'last_validated_step'):
                self.last_validated_step = -1

            # Freeze/unfreeze logic is handled in _init_model based on --no-freeze-backbone argument

            # Training loop
            self.model.train()

            batches_this_epoch = 0

            # Make epoch finite even with replacement samplers
            train_iter = iter(train_loader)
            max_batches = optim_steps_per_epoch * self.args.accum_steps
            for step in range(1, max_batches + 1):
                try:
                    chunks, labels, mask, spacing = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    chunks, labels, mask, spacing = next(train_iter)
                batches_this_epoch += 1
                # Monotonic global iteration across epochs
                if not hasattr(self, 'global_iter'):
                    self.global_iter = 0
                self.global_iter += 1
                t_fetch_start = time.time()
                fetch_ms = (t_fetch_start - self._last_batch_end) * 1000
                self.fetch_avg = beta * self.fetch_avg + (1.0 - beta) * fetch_ms
                t_compute_start = time.time()
                
                step_metrics = self._train_step(chunks, labels, mask, spacing)
                
                compute_ms = (time.time() - t_compute_start) * 1000
                self.compute_avg = beta * self.compute_avg + (1.0 - beta) * compute_ms
                self._last_batch_end = time.time()

                # Track if optimizer stepped in THIS iteration
                stepped_now = False

                # Gradient accumulation and optimizer step
                if step % self.args.accum_steps == 0:
                    # Manual linear warmup for the first warmup_steps using stored initial_lr
                    if self.global_step < self.args.warmup_steps:
                        warm_frac = (self.global_step + 1) / max(1, self.args.warmup_steps)
                        for pg in self.optimizer.param_groups:
                            base_lr = pg.get('initial_lr', pg['lr'])
                            pg['lr'] = base_lr * warm_frac

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Advance cosine scheduler only after warmup
                    if self.global_step >= self.args.warmup_steps:
                        self.scheduler.step()

                    # Increment global_step only on optimizer updates (after accum)
                    self.global_step += 1
                    stepped_now = True
                    if self.global_step == 1:
                        print("First real step → LR", self.scheduler.get_last_lr())

                    # Save checkpoint every 10k optimizer steps
                    if self.global_step % 10_000 == 0:
                        self.model_saver.save_checkpoint(
                            self.model,
                            epoch=self.current_epoch+1,
                            global_step=self.global_step
                        )

                # Print training metrics every print_every batches (per-iteration logging)
                if (
                    step_metrics
                    and self.global_iter % self.args.print_every == 0
                ):
                    print(f"\nIter {self.global_iter} (Epoch {self.current_epoch+1}):")
                    print(f"Loss: {step_metrics['loss']:.4f}")
                    print(f"Learning rate: {step_metrics['lr']:.6f}")
                    logging.info(
                        f"step={self.global_iter} epoch={self.current_epoch+1} loss={step_metrics['loss']:.4f} lr={step_metrics['lr']:.6f}"
                    )

                    line = (f"Step {self.global_iter:>7}  "
                            f"Ep {self.current_epoch+1:02d}  "
                            f"loss {step_metrics['loss']:.4f}  "
                            f"lr {step_metrics['lr']:.6f}  "
                            f"fetch {self.fetch_avg:6.1f} ms  "
                            f"compute {self.compute_avg:6.1f} ms")
                    print("\n" + line)
                    logging.info(line)

                    # Print individual group metrics (Acc only during training)
                    for g in self.group_names:
                        ak = f'acc_{g}_running'
                        if ak in step_metrics:
                            acc_val = step_metrics[ak]
                            print(f"  Group {g}:  Acc {acc_val:.3f}   (Sens/Spec calculated in validation)")
                            logging.info(f"group={g} acc_running={acc_val:.4f}")

                    # Print individual binary metrics (Acc, Sens, Spec, Bal_Acc)
                    for name in self.binary_names:
                        ak  = f'acc_{name}_running'
                        sk  = f'sens_{name}_running'
                        spk = f'spec_{name}_running'
                        if ak in step_metrics:
                            acc_val = step_metrics[ak]
                            sens_val = step_metrics.get(sk, float('nan'))
                            spec_val = step_metrics.get(spk, float('nan'))
                            bal_acc = (sens_val + spec_val) / 2 if not (np.isnan(sens_val) or np.isnan(spec_val)) else float('nan')
                            print(f"  Binary {name}:  Acc {acc_val:.3f}   Sens {sens_val:.3f}   Spec {spec_val:.3f}   Bal_Acc {bal_acc:.3f}")
                            logging.info(
                                f"binary={name} acc_running={acc_val:.4f} sens_running={sens_val:.4f} spec_running={spec_val:.4f} bal_acc_running={bal_acc:.4f}")

                    # Log training metrics to file - only running metrics
                    train_metrics = {
                        "iteration": self.global_iter,
                        "epoch": epoch,
                        "loss": step_metrics['loss'],
                        "lr": step_metrics['lr'],
                        "type": "training"
                    }

                    # Log group metrics
                    for g in self.group_names:
                        acc_key = f'acc_{g}_running'
                        if acc_key in step_metrics:
                            train_metrics[f'acc_{g}'] = step_metrics[acc_key]
                    
                    # Log binary metrics
                    for name in self.binary_names:
                        acc_key = f'acc_{name}_running'
                        sens_key = f'sens_{name}_running'
                        spec_key = f'spec_{name}_running'
                        if acc_key in step_metrics:
                            train_metrics[f'acc_{name}'] = step_metrics[acc_key]
                        if sens_key in step_metrics:
                            train_metrics[f'sens_{name}'] = step_metrics[sens_key]
                        if spec_key in step_metrics:
                            train_metrics[f'spec_{name}'] = step_metrics[spec_key]

                    # Standardize epoch index in metrics to 1-based for display parity
                    train_metrics["epoch"] = self.current_epoch + 1
                    self.metrics_logger.log_metrics(train_metrics)

                    # Mark this step as logged
                    self.last_logged_step = self.global_step

                # (global_step now increments only on optimizer updates)

                # Validation per iteration (batch) at val_every interval
                if (
                    val_loader
                    and self.global_iter % self.args.val_every == 0
                ):
                    print(f"\nReached validation point at iter {self.global_iter} (epoch {epoch+1})")

                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                    self._run_validation(val_loader)

                    gc.collect()
                    torch.cuda.empty_cache()


            # --- FLUSH PARTIAL ACCUMULATION (if epoch ended mid-accum) ---
            # Count how many mini-batches we actually processed this epoch
            remainder = batches_this_epoch % self.args.accum_steps if batches_this_epoch > 0 else 0
            if remainder != 0:
                # Manual linear warmup for the first warmup_steps using stored initial_lr
                if self.global_step < self.args.warmup_steps:
                    warm_frac = (self.global_step + 1) / max(1, self.args.warmup_steps)
                    for pg in self.optimizer.param_groups:
                        base_lr = pg.get('initial_lr', pg['lr'])
                        pg['lr'] = base_lr * warm_frac

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Advance cosine scheduler only after warmup
                if self.global_step >= self.args.warmup_steps:
                    self.scheduler.step()

                # Increment global_step only on optimizer updates (after accum/flush)
                self.global_step += 1
                stepped_now = True

                # (optional) run validation/checkpoint if they align with this step
                if (
                    val_loader 
                    and self.global_step % self.args.val_every == 0 
                    and self.global_step != self.last_validated_step
                ):
                    self._run_validation(val_loader)
                    self.last_validated_step = self.global_step

        # Save final model
        checkpoint_metadata = {
            'label_cols': self.label_cols,
            'model_config': {
                'num_attn_heads': self.args.num_attn_heads,
                'num_layers': self.args.num_layers,
                'dropout_rate': self.args.dropout
            },
            'final_metrics': {
                'total_steps': self.global_step,
                'epochs_completed': epochs
            }
        }
        self.model_saver.save_checkpoint(
            model=self.model,
            epoch=epochs + 1,
            global_step=self.global_step,
            metadata=checkpoint_metadata,
            is_final=True
        )



    def _run_validation(self, val_loader):
        """Run validation and log results"""
        validation_start_time = time.time()

        print(f"\n=== Starting validation at step {self.global_step} ===")

        metrics = self.evaluate(val_loader)

        elapsed = time.time() - validation_start_time

        print(f"Validation completed in {elapsed:.2f} seconds")

        val_metrics = metrics.copy()
        val_metrics.update({
            "iteration": getattr(self, 'global_iter', -1),
            "epoch": self.current_epoch + 1,
            "lr": self.scheduler.get_last_lr()[0],
            "validation_time_seconds": elapsed,
            "type": "validation"
        })
        self.metrics_logger.log_metrics(val_metrics)

        print("Per‐task validation results:")
        # Print group results - individual categories (Acc, Sens, Spec, Bal_Acc)
        for g in self.group_names:
            acc = val_metrics.get(f'acc_{g}', float('nan'))
            # Use macro-averaged metrics for display
            sens = val_metrics.get(f'macro_sens_{g}', float('nan'))
            spec = val_metrics.get(f'macro_spec_{g}', float('nan'))
            bal_acc = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else float('nan')
            print(f"  Group {g}:  Acc {acc:.3f}   Sens {sens:.3f}   Spec {spec:.3f}   Bal_Acc {bal_acc:.3f}")
        
        # Print binary results - individual categories (AUC, Acc, Sens, Spec, Bal_Acc)
        for name in self.binary_names:
            auc = val_metrics.get(f'auc_{name}',  float('nan'))
            acc = val_metrics.get(f'acc_{name}',  float('nan'))
            sens = val_metrics.get(f'sens_{name}', float('nan'))
            spec = val_metrics.get(f'spec_{name}', float('nan'))
            bal_acc = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else float('nan')
            print(f"  Binary {name}:  AUC {auc:.3f}   Acc {acc:.3f}   Sens {sens:.3f}   Spec {spec:.3f}   Bal_Acc {bal_acc:.3f}")

        print(f"=== Validation complete at step {self.global_step} ===\n")

    def reset_running_metrics(self):
        """Reset running metrics for a new epoch"""
        # Initialize group metrics
        self.correct_preds = {f'grp_{g}': 0 for g in self.group_names}
        self.total_preds = {f'grp_{g}': 0 for g in self.group_names}
        
        # Initialize binary metrics
        for name in self.binary_names:
            tkey = f'bin_{name}'
            self.correct_preds[tkey] = 0
            self.total_preds[tkey] = 0
            self.tp_preds[tkey] = 0
            self.fn_preds[tkey] = 0
            self.tn_preds[tkey] = 0
            self.fp_preds[tkey] = 0


def log_all_arguments(args):
    """Log all command line arguments to the training log"""
    logging.info("=" * 80)
    logging.info("TRAINING ARGUMENTS")
    logging.info("=" * 80)
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        logging.info(f"{arg_name:25}: {arg_value}")
    logging.info("=" * 80)


def main(args):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output, 'training.log'))
        ],
        force=True
    )
    logging.info("Starting single-GPU training")

    log_all_arguments(args)

    if args.balance_val:
        csv_path = prepare_balanced_validation(args.csv)
    else:
        csv_path = args.csv

    processor = VolumeProcessor(chunk_depth=3, out_size=(448, 448))

    # Parse groups and auto-generate label-cols
    try:
        parsed_groups = parse_groups_list(args.groups)
        # Flatten groups and combine with binary names, removing duplicates
        label_cols = list(dict.fromkeys([label for group in parsed_groups for label in group] + args.binary_names))
        print(f"Auto-generated label-cols: {label_cols}")
    except ValueError as e:
        print(f"Error parsing groups: {e}")
        print("Expected format: '[[a,b,c], [d], [e,f,g]]'")
        print("Example: '[[\"is_scc\", \"is_adenocarcinoma\", \"is_other\"], [\"is_M0\", \"is_M1A\"]]'")
        return

    balanced_col = label_cols[0]
    split_col = args.split_col
    full_df = pd.read_csv(csv_path)

    train_ds = NLSTDataset(
        df=full_df[full_df[split_col] == "train"],
        processor=processor,
        label_cols=label_cols,
        augment=augment_3_channel
    )
    print("Train dataset created")
    print_training_dataset_stats(csv_path, label_cols, split_col)

    if args.balanced_sampling == "True":
        train_sampler = make_balanced_sampler(train_ds, balanced_col)
        print(f"Using balanced sampling based on column: {balanced_col}")
        logging.info(f"Using balanced sampling based on column: {balanced_col}")
    else:
        train_sampler = None
        print("Using standard sampling with shuffle")
        logging.info("Using standard sampling with shuffle")

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size= 1, #args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        pin_memory=True,
    )

    full_df = pd.read_csv(csv_path)
    val_df = full_df[full_df[split_col] == "val"].copy()

    print(f"Total validation samples: {len(val_df)}")

    val_ds = NLSTDataset(
        df=val_df,
        processor=processor,
        label_cols=label_cols
    )

    val_loader = DataLoader(
        val_ds,
        batch_size= 1, #args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0]
    )

    print(f"Created validation loader with {len(val_loader)} samples")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to dictionary format with provided group names
    groups_def = {}
    for i, group_labels in enumerate(parsed_groups):
        if i < len(args.group_names):
            group_name = args.group_names[i]
        else:
            group_name = f"group_{i+1}"
        groups_def[group_name] = group_labels
    
    print(f"\nGroup Definitions:")
    for group_name, group_labels in groups_def.items():
        print(f"  {group_name}: {group_labels}")
    print(f"Binary Names: {args.binary_names}")
    logging.info(f"Group Definitions: {groups_def}")
    logging.info(f"Binary Names: {args.binary_names}")
    
    trainer = SingleGPUTrainer(args, label_cols, groups_def, args.binary_names)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,

    )

    checkpoint_metadata = {
        'label_cols': label_cols,
        'model_config': {
            'num_attn_heads': args.num_attn_heads,
            'num_layers': args.num_layers,
            'dropout_rate': args.dropout
        },
        'final_metrics': {
            'total_steps': trainer.global_step,
            'epochs_completed': args.epochs
        }
    }
    trainer.model_saver.save_checkpoint(
        model=trainer.model,
        epoch=args.epochs,
        global_step=trainer.global_step,
        metadata=checkpoint_metadata,
        is_final=True
    )

    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())

    print(f"\n==== MODEL PARAMETERS ====")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print(f"\n==== GPU MEMORY USAGE ====")
        print(f"Allocated: {allocated:.2f} MB")
        print(f"Reserved: {reserved:.2f} MB")
        print(f"Max Allocated: {max_allocated:.2f} MB")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU Training with Learnable Task Weights")
    parser.add_argument("--csv", type=str, default="/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/CSVs/ready_finetune_t1258_v330.csv",
                        help="Path to the CSV file containing dataset information")
    
    # Group definitions for mutually exclusive tasks
    parser.add_argument("--groups", type=str,
                       default="[['is_scc', 'is_adenocarcinoma', 'is_other'], ['is_M0', 'is_M1A', 'is_M1B', 'is_M1C'], ['is_N0', 'is_N1', 'is_N2', 'is_N3'], ['is_T0', 'is_T1', 'is_T2', 'is_T3', 'is_T4']]",
                       help="List of groups in Python list format. Each inner list contains mutually exclusive labels that sum to 1. Format: '[[a,b,c], [d], [e,f,g]]'")
    
    # Group names for the prediction tasks
    parser.add_argument("--group-names", type=str, nargs='+', 
                       default=['metastasis', 'node', 'tumor'],
                       help="Names for the groups (must match number of groups)")
    
    # Binary (independent) task names
    parser.add_argument("--binary-names", type=str, nargs='+', 
                       default=[],
                       help="Independent binary task names")
    parser.add_argument("--balance-val", action="store_true",
                        help="Whether to balance the validation set")
    parser.add_argument("--class-weights", type=str, default="True", choices=["True", "False"],
                        help="Whether to use class weights in the loss function (True/False, default: True)")
    parser.add_argument("--balanced-sampling", type=str, default="False", choices=["True", "False"],
                        help="Whether to use balanced sampling for training (True/False, default: False)")
    parser.add_argument("--accum-steps", type=int, default=100,
                        help="Number of steps to accumulate gradients over")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Actual batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train for")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Number of warmup steps for learning rate scheduling")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"],
                        help="Optimizer to use")
    parser.add_argument("--num-attn-heads", type=int, default=3,
                        help="Number of attention heads in the transformer aggregator")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of layers in the transformer aggregator")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in the transformer aggregator")
    parser.add_argument("--output", type=str, default="./output_single_gpu_learnable",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--print-every", type=int, default=100,
                        help="Print training stats every N steps")
    parser.add_argument("--val-every", type=int, default=500,
                        help="Run validation every N steps")
    parser.add_argument("--metrics-dir", type=str,
                        default="/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/output/metrics",
                        help="Directory to save training metrics")
    parser.add_argument("--max-chunks", type=int, default=64, help="Maximum number of chunks to process per sample")
    parser.add_argument("--no-load-pretrained", action="store_true", default=False,
                        help="Disable loading pretrained weights (use random initialization)")
    parser.add_argument("--no-freeze-backbone", action="store_true",
                        help="Disable freezing the DINO backbone (train all parameters) - DEFAULT: backbone is frozen")
    parser.add_argument("--no-learnable-weights", action="store_true",
                        help="Disable learnable task weights (use equal weights) - DEFAULT: learnable weights are enabled")
    parser.add_argument("--loss-wrapper", type=str, choices=["multitask"], default="multitask",
                        help="Use UncertaintyWeightedSumNamed for learnable task weights (default: enabled)")
    parser.add_argument("--split-col", type=str, default="split", help="Name of the column that marks the train/val rows.")
    parser.add_argument("--lse-tau", type=float, default=1.0,
                        help="Temperature parameter for LSE (Log-Sum-Exp) aggregation (default: 1.0)")
    parser.add_argument("--waqas-way", action="store_true",
                        help="Use Waqas way for attention aggregation (default: False)")
    parser.add_argument("--use-focal-loss", action="store_true", default=False,
                        help="Whether to use focal loss instead of cross entropy for group tasks")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Check if we should install packages (only first time)
    if not os.path.exists(os.path.join(args.output, '.packages_installed')):
        install_packages()
        with open(os.path.join(args.output, '.packages_installed'), 'w') as f:
            f.write('Packages installed\n')

    main(args)
