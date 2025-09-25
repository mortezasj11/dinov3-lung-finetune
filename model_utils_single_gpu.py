# File: model_utils_single_gpu.py
# Single-GPU compatible version of model utilities

import os, re, glob, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import json
import time
import subprocess
from torch.utils.data import WeightedRandomSampler
from torchmetrics.functional import auroc
from sklearn.metrics import roc_auc_score
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Callable
import torchvision.transforms as T
import sys
from typing import Tuple
sys.path.insert(0, "/rsrch1/ip/msalehjahromi/codes/dinov2-torchrun-dataloader6")
from dinov2.models.vision_transformer import DinoVisionTransformer

def focal_ce(logits, target, alpha=None, gamma=2.0):
    """Focal Cross Entropy Loss for handling class imbalance"""
    ce = F.cross_entropy(logits, target, weight=alpha, reduction='none')
    pt = torch.gather(F.softmax(logits, dim=-1), 1, target.unsqueeze(1)).squeeze(1)
    return ((1-pt)**gamma * ce).mean()

def install_packages():
    commands = [
        ["pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu117", "torch==2.0.0", "torchvision==0.15.0", "omegaconf", "torchmetrics==0.10.3", "fvcore", "iopath", "xformers==0.0.18", "submitit","numpy<2.0"],
        ["pip", "install", "--extra-index-url", "https://pypi.nvidia.com", "cuml-cu11"],
        ["pip", "install", "black==22.6.0", "flake8==5.0.4", "pylint==2.15.0"],
        ["pip", "install", "mmsegmentation==0.27.0"],
        ["pip", "install", "mmcv-full==1.5.0"]
    ]
    for i, command in enumerate(commands):
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def prepare_balanced_validation(csv_path):
    """Prepare a balanced validation set from the original data"""
    df = pd.read_csv(csv_path)
    
    # Print original validation set size
    print(f"Original validation set size: {df[df['split']=='val'].shape}")

    val_df = df[df.split == 'val'].copy()
    val_df = val_df[val_df.file_path.notna()].reset_index(drop=True)

    # Get positive and negative samples for 6-year-cancer
    pos_samples = val_df[val_df['6-year-cancer'] == 1]
    neg_samples = val_df[val_df['6-year-cancer'] == 0]

    print("\nBefore balancing:")
    print(f"Positive samples: {len(pos_samples)}")
    print(f"Negative samples: {len(neg_samples)}")

    # Calculate sizes for balanced set
    n_pos = len(pos_samples)
    n_neg = len(neg_samples)
    target_size = min(n_pos, n_neg)

    # Sample equally from positive and negative
    balanced_pos = pos_samples.sample(n=target_size, random_state=42)
    balanced_neg = neg_samples.sample(n=target_size, random_state=42)

    # Combine balanced samples
    balanced_val_df = pd.concat([balanced_pos, balanced_neg]).reset_index(drop=True)

    print("\nAfter balancing:")
    print(f"Balanced validation set shape: {balanced_val_df.shape}")

    # Create new dataframe correctly
    df_new = df.copy()
    # Remove all validation samples
    df_new = df_new[df_new.split != 'val']
    # Add balanced validation samples
    balanced_val_df['split'] = 'val'  # Ensure split column is set
    df_new = pd.concat([df_new, balanced_val_df], ignore_index=True)

    print("\nFinal shapes:")
    print(f"New validation set shape: {df_new[df_new['split']=='val'].shape}")
    print(f"Total dataset shape: {df_new.shape}")

    # Save the balanced validation set
    temp_csv_path = csv_path.replace('.csv', '_balanced.csv')
    df_new.to_csv(temp_csv_path, index=False)
    
    return temp_csv_path

def calculate_auc(predictions, targets, mask=None):
    """Calculate AUC score for binary classification"""
    if mask is not None:
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return float('nan')
        predictions = predictions[valid_indices]
        targets = targets[valid_indices]
    
    # Ensure we have both classes present
    if np.all(targets == 1) or np.all(targets == 0):
        return float('nan')
    
    try:
        return roc_auc_score(targets, predictions)
    except:
        return float('nan')

class MetricsLogger:
    def __init__(self, rank: int = 0, metrics_dir: str = None):
        # For single-GPU, rank is always 0
        self.rank = 0
        self.should_save_metrics = (metrics_dir is not None)
        
        if self.should_save_metrics:
            self.metrics_dir = metrics_dir
            os.makedirs(self.metrics_dir, exist_ok=True)
            
            # Create metrics filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.metrics_filename = os.path.join(
                self.metrics_dir, 
                f"training_metrics_single_gpu_{timestamp}.jsonl"
            )
            
            print(f"Will save metrics to: {self.metrics_filename}")
    
    def log_metrics(self, metrics_dict: Dict[str, Any]):
        """Log metrics to file and console"""
        if not self.should_save_metrics:
            return
            
        # Log to console for both training and validation metrics
        log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                            for k, v in metrics_dict.items()])
        import logging
        logging.info(log_str)
        
        # Always save to file
        try:
            with open(self.metrics_filename, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
        except Exception as e:
            print(f"ERROR saving metrics: {e}")

class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.samples_seen = 0
        
        # Accuracy tracking
        self.correct_1yr = 0
        self.correct_3yr = 0
        self.correct_6yr = 0
        
        # Sample counting
        self.total_1yr_samples = 0
        self.total_3yr_samples = 0
        self.total_6yr_samples = 0
        
        # Positive sample tracking
        self.pos_1yr = 0
        self.pos_3yr = 0
        self.pos_6yr = 0
    
    def update_metrics(self, predictions, targets, mask, loss):
        """Update metrics for a single batch"""
        # Update loss
        self.total_loss += loss
        self.samples_seen += 1
        
        # Update accuracy and counts for each time point
        if mask[0]:  # 1-year
            is_correct = (predictions[0] == targets[0]).float().item()
            self.correct_1yr += is_correct
            self.total_1yr_samples += 1
            self.pos_1yr += int(targets[0].item() == 1)
            
        if mask[1]:  # 3-year
            is_correct = (predictions[1] == targets[1]).float().item()
            self.correct_3yr += is_correct
            self.total_3yr_samples += 1
            self.pos_3yr += int(targets[1].item() == 1)
            
        if mask[2]:  # 6-year
            is_correct = (predictions[2] == targets[2]).float().item()
            self.correct_6yr += is_correct
            self.total_6yr_samples += 1
            self.pos_6yr += int(targets[2].item() == 1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return current metrics"""
        metrics = {
            "total_loss": self.total_loss / max(1, self.samples_seen),
            "acc1": self.correct_1yr / max(1, self.total_1yr_samples),
            "acc3": self.correct_3yr / max(1, self.total_3yr_samples),
            "acc6": self.correct_6yr / max(1, self.total_6yr_samples),
            "pos_count_1yr": f"{self.pos_1yr}-{self.total_1yr_samples - self.pos_1yr}",
            "pos_count_3yr": f"{self.pos_3yr}-{self.total_3yr_samples - self.pos_3yr}",
            "pos_count_6yr": f"{self.pos_6yr}-{self.total_6yr_samples - self.pos_6yr}"
        }
        return metrics

class MILAttentionPool(nn.Module):
    """
    Enhanced attention pooling with slightly bigger architecture.
    Input:  x  [B, S, D]
    Output: z  [B, D]  pooled embedding
            a  [B, S]  attention weights
    """
    def __init__(self, in_dim: int = 771, hidden_dim: int = 1024):
        super().__init__()
        # First layer: input to hidden
        self.v1 = nn.Linear(in_dim, hidden_dim)
        # Second layer: hidden to hidden (slightly smaller)
        self.v2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Attention weights layer
        self.w = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h1 = torch.tanh(self.v1(x))            # [B,S,H]
        h1 = self.dropout(h1)                  # Apply dropout
        h2 = torch.tanh(self.v2(h1))           # [B,S,H/2]
        a = self.w(h2).squeeze(-1)             # [B,S]
        if mask is not None:
            a = a.masked_fill(~mask.bool(), float('-inf'))
        a = F.softmax(a, dim=1)                # [B,S]
        z = torch.bmm(a.unsqueeze(1), x).squeeze(1)   # [B,D]
        return z, a
    
# class CombinedModel(nn.Module):
#     def __init__(
#         self,
#         base_model,
#         chunk_feat_dim: int = 768,
#         hidden_dim: int = 1024,
#         num_tasks: int = 1,
#         num_attn_heads: int = 8,
#         num_layers: int = 2,
#         dropout_rate: float = 0.2,
#         lse_tau: float = 1.0,
#         waqas_way: bool = False,
#     ):
#         super().__init__()
#         self.base = base_model      # now parameters of the base model are referenced
#         self.lse_tau = lse_tau
#         self.waqas_way = waqas_way
#         self.chunk_feat_dim = chunk_feat_dim + 3   # 768 + 3 = 771

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.chunk_feat_dim,
#             nhead=num_attn_heads,
#             dim_feedforward=hidden_dim,
#             dropout=dropout_rate,
#             batch_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_layers
#         )
#         self.attn = MILAttentionPool(in_dim=771, hidden_dim=1024)
#         self.classifier = nn.Linear(self.chunk_feat_dim, num_tasks)
#         #self.waqas_way = True

#         # ─── shorter _augment_feats ─────────────────────────────────────────────
#     def _augment_feats(self, feats: torch.Tensor) -> torch.Tensor:
#         if not self.training:                                   # keep eval intact
#             return feats
#         S, dev = feats.size(0), feats.device
#         if torch.rand(1, device=dev) < 0.5:                     # reverse
#             feats = feats.flip(0)
#         if S > 1 and torch.rand(1, device=dev) < 0.9:           # circular roll
#             feats = torch.roll(feats, torch.randint(1, S, (1,), device=dev).item(), 0)
#         if torch.rand(1, device=dev) < 0.6:                     # random shuffle
#             feats = feats[torch.randperm(S, device=dev)]
#         return feats


#     #@torch.no_grad()          # remove this decorator if you want the base to train
#     def _chunk_embed(self, chunk):
#         return self.base(chunk.unsqueeze(0))       # → [1, 768]

#     def forward(self, x: torch.Tensor, spacing=(1.0, 1.0, 1.0)):
#         """
#         x:       [S, C, H, W]  – S chunks for a single patient
#         spacing: (dx, dy, dz)  – voxel size in mm (or any scale units)
#         """
#         S = x.size(0)
#         device = x.device
#         dtype  = x.dtype
#         feats = torch.cat([self._chunk_embed(x[i]) for i in range(S)], dim=0)  # [S, 768]
#         spacing_vec = torch.tensor(spacing, dtype=dtype, device=device).expand(S, 3)
#         feats = torch.cat([feats, spacing_vec], dim=1)                        # [S, 771]
#         feats = feats[torch.sort(torch.randperm(S:=feats.size(0), device=feats.device)[int(0.15*S):])[0]] # just added July 5; from .35 to .15 in sep 20
#         feats = self._augment_feats(feats)                    # optional reorder

#         if  self.waqas_way:
#             pooled, a1 = self.attn(feats.unsqueeze(0))   # [1,D], [1,S]
#             return self.classifier(pooled) 
        
#         encoded = self.transformer(feats.unsqueeze(0))  # [1, S, 771]
#         pooled = self.lse_tau * torch.logsumexp(encoded / self.lse_tau, dim=1)  # [1, 771]
#         return self.classifier(pooled)       # [1, num_tasks]


class VolumeProcessor:
    def __init__(self, chunk_depth=3, out_size=(448, 448), vmin=-1000, vmax=150, eps=0.00005):
        self.chunk_depth = chunk_depth
        self.out_size = out_size
        self.vmin = vmin
        self.vmax = vmax
        self.eps = eps
    
    def _preprocess_volume(self, npz_path):
        # Load NPZ file
        npz_data = np.load(npz_path)
        volume = npz_data['data']
        
        # Normalize to 0-1 range
        volume = (volume - volume.min()) / (volume.max() - volume.min() + self.eps)
        
        return volume
    
    def _get_chunks(self, volume):
        # Extract 3-slice chunks with stride 1
        depth, height, width = volume.shape
        
        # Create list of chunks
        chunks = []
        for i in range(0, depth - self.chunk_depth + 1):
            chunk = volume[i:i+self.chunk_depth]  # Get 3 consecutive slices
            
            # Resize if needed
            if (height, width) != self.out_size:
                chunk_resized = np.zeros((self.chunk_depth, *self.out_size))
                for j in range(self.chunk_depth):
                    # Use CV2 or other resize method here
                    # For simplicity, this is a placeholder
                    chunk_resized[j] = chunk[j]  # Replace with actual resize
                chunk = chunk_resized
            
            # Create "RGB" channels by duplicating (3D to pseudo-RGB)
            chunk_rgb = np.stack([chunk] * 3, axis=1)  # [3, 3, H, W]
            chunks.append(chunk_rgb)
        
        return chunks
    
    def process_file(self, npz_path):
        volume = self._preprocess_volume(npz_path)
        chunks = self._get_chunks(volume)
        return chunks


class RandomIntensityScaleShift:
    def __init__(self, scale=(0.98,1.02), shift=(-0.02,0.02), p=0.5):
        self.scale, self.shift, self.p = scale, shift, p

    def __call__(self, x: torch.Tensor):
        if random.random() < self.p:
            s = random.uniform(*self.scale)
            b = random.uniform(*self.shift)
            x = x * s + b
        return x.clamp(0, 1)   ####### -1, 1 if aug after scaling t = (t - 0.5) / 0.5 

# augment_3_channel = T.Compose([ # Sep 22, 2025, changed from fixed 0.35 to this
#     T.RandomResizedCrop((448, 448), scale=(0.7,1.3), ratio=(0.80,1.2), antialias=True),
#     T.RandomHorizontalFlip(),
#     T.RandomVerticalFlip(),
#     T.RandomRotation(90), 
#     T.RandomAffine(degrees=0, translate=(0.4,0.4), scale=(0.82,1.18)),
#     T.GaussianBlur(kernel_size=5, sigma=(0.1,1.0)),
#     RandomIntensityScaleShift(scale=(0.82,1.18), shift=(-0.08,0.08), p=0.8),
# ])

augment_3_channel = T.Compose([    # Sep 22, 2025, changed 
    T.RandomResizedCrop((448, 448), scale=(0.8, 1.2), ratio=(0.8, 1.2), antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(25),
    T.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
    T.Lambda(lambda x: (x.clamp(0,1) ** np.random.uniform(0.8, 1.2)).clamp(0,1)),     # Gamma + stronger noise
    T.Lambda(lambda x: (x + torch.randn_like(x) * 0.05).clamp(0,1)),
    T.RandomErasing(p=0.4, scale=(0.02, 0.12), ratio=(0.4, 2.5), value=0.0, inplace=True),
    RandomIntensityScaleShift(scale=(0.8, 1.20), shift=(-0.15, 0.15), p=0.95),
])

class NLSTDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 processor: VolumeProcessor,
                 label_cols: List[str],
                 augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        df = df[df.file_path.notna()].reset_index(drop=True)
        self.df = df
        self.proc = processor
        self.labels = label_cols
        self.augment = augment
        print(f"Augmentation: {self.augment}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]  # Shape: pandas.Series (1D row data)
        nii_path = row.file_path  # Shape: str (file path)    /rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology//FineTune/ready_finetune_max192/CT_Main_0002.nii.gz
        #print(f"[Data] Loading volume {idx}: {nii_path}")

        # becomes: /rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/ready_finetune_max192_npz/CT_Main_0002.npz
        p = row.file_path
        # Replace the directory name and file extension
        npz_path = p.replace('ready_finetune_max192', 'ready_finetune_max192_npz').replace('.nii.gz', '.npz')
        
        # Convert to shared path - handle slashes properly
        #if inside k8s and shared is mounted
        if os.path.exists("/shared"):
            if os.path.exists("/shared/ready_finetune_max192_npz"):   
                npz_path = p.replace('ready_finetune_max192', 'ready_finetune_max192_npz').replace('.nii.gz', '.npz')
                OLD_ROOT = "/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/ready_finetune_max192_npz"
                rel = os.path.relpath(npz_path, OLD_ROOT)
                # if the initContainer copied the dir itself, it's under /shared/ready_finetune_max192_npz/...
                shared_root = "/shared/ready_finetune_max192_npz" if os.path.isdir("/shared/ready_finetune_max192_npz") else "/shared"
                npz_path = os.path.join(shared_root, rel)
            elif os.path.exists("/shared/ready_finetune_max192_npz_mnt"):
                npz_path = p.replace('ready_finetune_max192', 'ready_finetune_max192_npz_mnt').replace('.nii.gz', '.npz')
                OLD_ROOT = "/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/ready_finetune_max192_npz_mnt"
                rel = os.path.relpath(npz_path, OLD_ROOT)
                # if the initContainer copied the dir itself, it's under /shared/ready_finetune_max192_npz_mnt/...
                shared_root = "/shared/ready_finetune_max192_npz_mnt" if os.path.isdir("/shared/ready_finetune_max192_npz_mnt") else "/shared"
                npz_path = os.path.join(shared_root, rel)

        # Debug: print the path conversion
        # if idx < 3:  # Only print for first few samples to avoid spam
        #     print(f"DEBUG: Original path: {p}")
        #     print(f"DEBUG: NPZ path: {npz_path}")
        # Add error handling for corrupted NPZ files
        try:
            # Load NPZ file instead of NIfTI
            npz_data = np.load(npz_path)
            vol = npz_data['data'].astype(np.float32)  # Shape: [H, W, D] 3D volume
            spacing = tuple(npz_data['spacing'])  # Get voxel spacing (dx, dy, dz)
        except Exception as e:
            print(f"Error loading file {npz_path}: {e}")
            raise e
            # Return a fallback sample with zeros
            # Create a dummy volume with the expected shape
            # dummy_vol = np.zeros((448, 448, 3), dtype=np.float32)  # Minimal 3D volume
            # chunks = torch.from_numpy(dummy_vol).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 448, 448]
            # labels = row[self.labels].to_numpy(dtype=np.float32)
            # mask = (labels != -1)
            # spacing = (1.0, 1.0, 1.0)  # Default spacing
            # return chunks, labels, mask, spacing
        H, W, D = vol.shape
        #print(f"[Data] Volume shape: {vol.shape}")
        num_chunks = D // self.proc.chunk_depth  # Shape: scalar (number of chunks)

        windows = []  # Will collect chunks
        for i in range(num_chunks):
            arr = vol[:, :, i*self.proc.chunk_depth:(i+1)*self.proc.chunk_depth]  # Shape: [H, W, chunk_depth]
            arr = np.clip(arr, self.proc.vmin, self.proc.vmax)  # Shape: [H, W, chunk_depth]
            arr = (arr - self.proc.vmin) / (self.proc.vmax - self.proc.vmin)  # Shape: [H, W, chunk_depth]
            arr = np.clip(arr, self.proc.eps, 1.0 - self.proc.eps)  # Shape: [H, W, chunk_depth]

            t = torch.from_numpy(arr).permute(2, 0, 1)  # Shape: [chunk_depth, H, W]
            t = F.interpolate(t.unsqueeze(0), size=self.proc.out_size,
                              mode="bilinear", align_corners=False).squeeze(0)  # Shape: [chunk_depth, 448, 448]
            if self.augment is not None:
                t = self.augment(t)
            t = (t - 0.5) / 0.5  # Shape: [chunk_depth, 448, 448]
            windows.append(t)  # Add to list

        # if self.augment is not None:
        #     windows = [self.augment(c) for c in windows]
        chunks = torch.stack(windows, dim=0)  # Shape: [num_chunks, chunk_depth, 448, 448]
        labels = row[self.labels].to_numpy(dtype=np.float32)  # Shape: [num_labels] (e.g., [3] for 3 time points)
        mask = (labels != -1)  # Shape: [num_labels] boolean mask
        return chunks, labels, mask, spacing  # Return shapes: [num_chunks, chunk_depth, 448, 448], [num_labels], [num_labels], (dx, dy, dz)

def calculate_class_weights(df, label_cols):
    """Calculate class weights for imbalanced binary labels (only 0 vs 1, ignore -1)."""
    weights = []
    for col in label_cols:
        # Only keep 0/1 entries
        vals = df[col].isin([0, 1])
        pos_count = (df.loc[vals, col] == 1).sum()
        neg_count = (df.loc[vals, col] == 0).sum()
        # If both classes exist, weight positive by neg/pos; else fallback to 1.0
        if pos_count > 0 and neg_count > 0:
            w = neg_count / pos_count
        else:
            w = 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

def print_dataset_statistics(df, label_cols, dataset_name="Dataset"):
    """
    Print detailed statistics about a dataset including class distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset
        label_cols (List[str]): List of column names for prediction tasks
        dataset_name (str): Name to display for this dataset
    """
    print(f"\n====== {dataset_name} Statistics ======")
    print(f"Total samples: {len(df)}")
    
    # Check if 'split' column exists
    if 'split' in df.columns:
        for split_name in df['split'].unique():
            split_count = (df['split'] == split_name).sum()
            print(f"  '{split_name}' split: {split_count} samples ({100 * split_count / len(df):.1f}%)")
    
    # Calculate class distribution for each time point
    print(f"\nClass distribution in {dataset_name}:")
    for col in label_cols:
        pos = (df[col] == 1).sum()
        neg = (df[col] == 0).sum()
        missing = (df[col] == -1).sum()
        invalid = len(df) - pos - neg - missing
        total_valid = pos + neg
        
        if total_valid > 0:
            pos_ratio = 100 * pos / total_valid
            print(f"  {col}:")
            print(f"    Positive: {pos} ({pos_ratio:.1f}%)")
            print(f"    Negative: {neg} ({100 - pos_ratio:.1f}%)")
            if missing > 0:
                print(f"    Missing: {missing} ({100 * missing / len(df):.1f}% of total)")
            if invalid > 0:
                print(f"    Invalid values: {invalid}")
        else:
            print(f"  {col}: No valid samples")
    
    # Check for file_path if present
    if 'file_path' in df.columns:
        missing_paths = df['file_path'].isna().sum()
        if missing_paths > 0:
            print(f"\nWarning: {missing_paths} samples ({100 * missing_paths / len(df):.1f}%) have missing file paths")
    
    # Volume information if available
    if 'vol_shape' in df.columns:
        print("\nVolume statistics:")
        try:
            # Extract volume shapes and calculate statistics
            shapes = df['vol_shape'].dropna().tolist()
            if shapes:
                print(f"  Number of volumes with shape info: {len(shapes)}")
                # Add any specific volume analysis you need
        except:
            print("  Could not analyze volume shapes")
    
    print("=" * (23 + len(dataset_name)))
    return

def analyze_datasets(csv_path, label_cols):
    """
    Analyze training and validation datasets from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        label_cols (List[str]): List of column names for prediction tasks
    """
    try:
        import pandas as pd
        
        # Read the full dataset
        full_df = pd.read_csv(csv_path)
        print_dataset_statistics(full_df, label_cols, "Full Dataset")
        
        # Analyze training set
        if 'split' in full_df.columns:
            train_df = full_df[full_df['split'] == 'train']
            if len(train_df) > 0:
                print_dataset_statistics(train_df, label_cols, "Training Set")
            
            # Analyze validation set
            val_df = full_df[full_df['split'] == 'val']
            if len(val_df) > 0:
                print_dataset_statistics(val_df, label_cols, "Validation Set")
                
            # Analyze test set if exists
            test_df = full_df[full_df['split'] == 'test']
            if len(test_df) > 0:
                print_dataset_statistics(test_df, label_cols, "Test Set")
        
        # Calculate imbalance ratios for potential class weights
        print("\n===== Class Weight Recommendations =====")
        for col in label_cols:
            pos = (full_df[full_df['split'] == 'train'][col] == 1).sum()
            neg = (full_df[full_df['split'] == 'train'][col] == 0).sum()
            if pos > 0 and neg > 0:
                weight = neg / pos
                print(f"  {col}: Recommended pos_weight = {weight:.2f}")
        print("======================================")
        
    except Exception as e:
        print(f"Error analyzing datasets: {e}")
        import traceback
        traceback.print_exc()

class ModelSaver:
    """
    Utility class to handle model saving and checkpointing for single-GPU training.
    
    This class handles:
    1. Saving regular checkpoints at specified intervals
    2. Saving separate components of a model (base model and aggregator)
    3. Saving metadata about the checkpoint
    """
    
    def __init__(self, output_dir, rank=0):
        """
        Initialize the model saver.
        
        Args:
            output_dir (str): Directory to save checkpoints to
            rank (int): Process rank (always 0 for single-GPU)
        """
        self.rank = 0  # Always 0 for single-GPU
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Will save checkpoints to {self.checkpoint_dir}")
    
    def save_checkpoint(self, model, epoch, global_step, metadata=None, is_final=False, save_components=True):
        """
        Save model checkpoint.
        
        Args:
            model: The model to save (single-GPU model)
            epoch (int): Current epoch
            global_step (int): Current global step
            metadata (dict, optional): Additional metadata to save
            is_final (bool): Whether this is the final checkpoint
            save_components (bool): Whether to save model components separately
        """
        # No rank check needed for single-GPU
        
        # Determine checkpoint filename
        if is_final:
            checkpoint_name = 'model_final.pt'
        else:
            checkpoint_name = f'model_ep{epoch}_it{global_step}.pt'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save the combined model (single-GPU, no DDP wrapper)
        torch.save(
            model.state_dict(), 
            checkpoint_path
        )
        
        # Save components separately if requested
        if save_components:
            self._save_model_components(model, epoch, global_step, is_final)
        
        # Save metadata
        self._save_metadata(model, epoch, global_step, metadata, is_final)
        
        print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
    
    def _save_model_components(self, model, epoch, global_step, is_final=False):
        """
        Save base model and aggregator components separately.
        
        Args:
            model: The model to save components of
            epoch (int): Current epoch
            is_final (bool): Whether this is the final checkpoint
        """
        # Extract state dictionaries (single-GPU, no DDP wrapper)
        combined_state_dict = model.state_dict()
        
        # Create separate state dictionaries
        base_state_dict = {}
        aggregator_state_dict = {}
        
        for name, param in combined_state_dict.items():
            if name.startswith('base.'):
                base_state_dict[name.replace('base.', '')] = param
            else:
                aggregator_state_dict[name] = param
        
        # Determine component checkpoint filenames
        if is_final:
            base_checkpoint_name = 'base_model_final.pt'
            aggregator_checkpoint_name = 'aggregator_final.pt'
        else:
            base_checkpoint_name       = f'base_ep{epoch}_it{global_step}.pt'
            aggregator_checkpoint_name = f'aggregator_ep{epoch}_it{global_step}.pt'
        
        # Save base model
        torch.save(
            base_state_dict, 
            os.path.join(self.checkpoint_dir, base_checkpoint_name)
        )
        
        # Save aggregator
        torch.save(
            aggregator_state_dict, 
            os.path.join(self.checkpoint_dir, aggregator_checkpoint_name)
        )
        
        print(f"Saved model components to:")
        print(f"  Base: {base_checkpoint_name}")
        print(f"  Aggregator: {aggregator_checkpoint_name}")
    
    def _save_metadata(self, model, epoch, global_step, extra_metadata=None, is_final=False):
        """
        Save metadata about the checkpoint.
        
        Args:
            model: The model
            epoch (int): Current epoch
            global_step (int): Current global step
            extra_metadata (dict, optional): Additional metadata to save
            is_final (bool): Whether this is the final checkpoint
        """
        # Create basic metadata
        metadata = {
            'epoch': epoch,
            'global_step': global_step,
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'is_final': is_final,
        }
        
        # Add model configuration if available (single-GPU, no DDP wrapper)
        if hasattr(model, 'chunk_feat_dim'):
            metadata['model_config'] = {
                'chunk_feat_dim': getattr(model, 'chunk_feat_dim', 768),
                'hidden_dim': 1024,  # Use default value if attribute not present
                'num_tasks': getattr(model, 'classifier').out_features if hasattr(model, 'classifier') else 6,
            }
            
            # Add transformer configuration if available
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
                transformer = model.transformer
                metadata['model_config'].update({
                    'num_attn_heads': transformer.layers[0].self_attn.num_heads if hasattr(transformer.layers[0], 'self_attn') else 8,
                    'num_layers': len(transformer.layers),
                    'dropout_rate': transformer.layers[0].dropout.p if hasattr(transformer.layers[0], 'dropout') else 0.2,
                })
        
        # Add extra metadata if provided
        if extra_metadata:
            metadata.update(extra_metadata)
        
        # Save metadata
        metadata_filename = f'metadata_{"final" if is_final else f"epoch_{epoch}"}.json'
        with open(os.path.join(self.checkpoint_dir, metadata_filename), 'w') as f:
            json.dump(metadata, f, indent=2)

# def _latest(path, pat=r"model_epoch_(\d+)\.pt"):
#     files = [f for f in os.listdir(path) if re.match(pat, f)]
#     if not files: return None
#     latest = max(files, key=lambda f: int(re.match(pat, f).group(1)))
#     return os.path.join(path, latest), int(re.search(pat, latest).group(1))

# def load_latest_checkpoint(model, out_dir, device="cpu", rank=0):
#     ckpt_dir = os.path.join(out_dir, "checkpoints")
#     if not os.path.isdir(ckpt_dir):
#         if rank==0: print(f"[resume] no {ckpt_dir} — fresh run")
#         return None
#     ckpt_path, epoch = _latest(ckpt_dir) or (None, None)
#     if ckpt_path is None:
#         if rank==0: print(f"[resume] no model_epoch_*.pt — fresh run")
#         return None
#     if rank==0: print(f"[resume] Loading checkpoint {ckpt_path}")
#     sd = torch.load(ckpt_path, map_location=device)
#     tgt = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
#     tgt.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd, strict=False)
#     return epoch        # caller can log/decide what to do next

def _epoch(f): m=re.search(r'ep(\d+)', f); return int(m.group(1)) if m else None

def _latest(d):
    for pat in ('model_*.pt', 'aggregator_*.pt', 'base_*.pt'):
        c = glob.glob(os.path.join(d, pat))
        if c:
            p = max(c, key=os.path.getmtime)
            return p, _epoch(os.path.basename(p))
    return None

def load_latest_checkpoint(model, out_dir, device="cpu", rank=0):
    """Load the latest checkpoint for single-GPU training"""
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        import logging
        logging.info(f"[resume] no {ckpt_dir} — fresh run")
        return None
    ckpt = _latest(ckpt_dir) or (None, None)
    if ckpt[0] is None:
        import logging
        logging.info(f"[resume] no model_epoch_*.pt — fresh run")
        return None
    path, epoch = ckpt
    import logging
    logging.info(f"[resume] Loading checkpoint from: {path}")
    logging.info(f"[resume] Checkpoint epoch: {epoch}")
    logging.info(f"[resume] Checkpoint file size: {os.path.getsize(path) / (1024*1024):.2f} MB")
    sd = torch.load(path, map_location=device)
    # Single-GPU model, no DDP wrapper
    missing, _ = model.load_state_dict(sd.get("model", sd), strict=False)  # missing keys
    total = len(model.state_dict())                                         # total params
    loaded = total - len(missing)
    import logging
    logging.info(f"[resume] Loaded {loaded}/{total} params ({loaded/total*100:.1f}%)")
    if missing:
        logging.info(f"[resume] Missing keys: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    return epoch

def make_balanced_sampler(dataset, label_col="1-year-cancer"):
    """
    Returns a WeightedRandomSampler that draws 0 / 1 for `label_col`
    with equal chance.  -1 rows are ignored.  Works on a DataFrame
    or on an NLSTDataset (uses its .df).
    """
    df = dataset.df if hasattr(dataset, "df") else dataset           # ①
    msk = df[label_col] != -1                                        # valid rows
    y   = df.loc[msk, label_col].to_numpy()
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    w0, w1 = 1.0 / n0, 1.0 / n1                                      # ②
    weights = np.zeros(len(df), dtype=np.float32)
    weights[msk] = np.where(y == 1, w1, w0)                          # ③
    return WeightedRandomSampler(weights, num_samples=len(df), replacement=True)

def print_training_dataset_stats(csv_path, label_cols, split_col="split"):
    """
    Print basic dataset statistics for training script.
    
    Args:
        csv_path (str): Path to the CSV file
        label_cols (List[str]): List of column names for prediction tasks
    """
    try:
        # Load the full dataframe
        full_df = pd.read_csv(csv_path)
        train_df = full_df[full_df[split_col] == 'train']
        val_df_stats = full_df[full_df[split_col] == 'val']
        
        print("\n==== Dataset Statistics ====")
        print(f"Total samples: {len(full_df)}")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df_stats)}")
        
        # Print class distribution for each label column
        for col in label_cols:
            if col in train_df.columns:
                # Use basic sum() to count positive samples
                pos_train = sum(train_df[col] == 1)
                pos_val = sum(val_df_stats[col] == 1)
                
                # Calculate percentages
                train_pct = 100 * pos_train / len(train_df) if len(train_df) > 0 else 0
                val_pct = 100 * pos_val / len(val_df_stats) if len(val_df_stats) > 0 else 0
                
                print(f"\n{col}:")
                print(f"  Train positive: {pos_train} ({train_pct:.2f}%)")
                print(f"  Val positive: {pos_val} ({val_pct:.2f}%)")
        print("============================\n")
        
    except Exception as e:
        print(f"Error printing dataset statistics: {e}")



class CombinedModel_weight(nn.Module):
    def __init__(
        self,
        base_model,
        chunk_feat_dim: int = 768,
        hidden_dim: int = 1024,
        num_tasks: int = 1,
        num_attn_heads: int = 8,
        num_layers: int = 1,
        dropout_rate: float = 0.2,
        lse_tau: float = 1.0,
        waqas_way: bool = False,
    ):
        super().__init__()
        self.base = base_model
        self.lse_tau = lse_tau
        self.waqas_way = waqas_way
        # +3 channels for spacing vector
        self.chunk_feat_dim = chunk_feat_dim + 3  

        # Transformer aggregator
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.chunk_feat_dim,
            nhead=num_attn_heads,
            batch_first=True,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        self.classifier = nn.Linear(self.chunk_feat_dim, num_tasks)
        
        # Attention pooling for waqas_way
        self.attn = MILAttentionPool(in_dim=self.chunk_feat_dim, hidden_dim=1024)

    def _augment_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return feats
        S, dev = feats.size(0), feats.device
        if torch.rand(1, device=dev) < 0.5:
            feats = feats.flip(0)
        if S > 1 and torch.rand(1, device=dev) < 0.9:
            feats = feats.roll(torch.randint(1, S, (1,), device=dev).item(), 0)
        if torch.rand(1, device=dev) < 0.6:
            feats = feats[torch.randperm(S, device=dev)]
        return feats

    def _chunk_embed(self, chunk: torch.Tensor) -> torch.Tensor:
        # one-chunk → [1, 768]
        return self.base(chunk.unsqueeze(0))

    def forward(self, x: torch.Tensor, spacing=(1.0, 1.0, 1.0)) -> torch.Tensor:
        """
        Returns:
            logits: [1, num_tasks]
        """
        S = x.size(0)
        device, dtype = x.device, x.dtype

        # 1) embed chunks
        feats = torch.cat([self._chunk_embed(x[i]) for i in range(S)], dim=0)  # [S, 768]

        # 2) append spacing
        spacing_vec = torch.tensor(spacing, device=device, dtype=dtype).expand(S, 3)
        feats = torch.cat([feats, spacing_vec], dim=1)                         # [S, 771]

        # 3) random subset+augment (drop 35% only during training; keep all for eval)
        if self.training:
            r = torch.empty(1, device=device).uniform_(0.5, 1.0).item()   # Sep 22, 2025, changed from fixed 0.35 to this
            keep = torch.sort(torch.randperm(S, device=device)[int(r*S):])[0]
            feats = feats[keep]
        feats = self._augment_feats(feats)

        # 4) aggregation method
        if self.waqas_way:
            pooled, a1 = self.attn(feats.unsqueeze(0))   # [1,D], [1,S]
            logits = self.classifier(pooled) 
        else:
            # transformer + LSE pooling
            encoded = self.transformer(feats.unsqueeze(0))                          # [1, S, D]
            pooled  = self.lse_tau * torch.logsumexp(encoded / self.lse_tau, dim=1) # [1, D]
            logits = self.classifier(pooled)                                        # [1, num_tasks]

        return logits
    

# class NLSTDataset_catch(Dataset):
#     """
#     NLSTDataset with caching strategy for improved performance.
#     Based on the caching approach from the loader_subtyping example.
#     """
#     def __init__(self,
#                  df: pd.DataFrame,
#                  processor: VolumeProcessor,
#                  label_cols: List[str],
#                  augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
#                  use_cache: bool = True,
#                  cache_size_limit: Optional[int] = None):
        
#         df = df[df.file_path.notna()].reset_index(drop=True)
#         self.df = df
#         self.proc = processor
#         self.labels = label_cols
#         self.augment = augment
#         self.use_cache = use_cache
#         self.cache_size_limit = cache_size_limit
        
#         # Initialize cache
#         self.cache = [None] * len(self.df)
        
#         # Cache statistics
#         self.cache_stats = {
#             'hits': 0,
#             'misses': 0,
#             'size': 0,
#             'memory_usage': 0
#         }
        
#         print(f"NLSTDataset_catch initialized:")
#         print(f"  Total samples: {len(self.df)}")
#         print(f"  Caching: {'ENABLED' if use_cache else 'DISABLED'}")
#         print(f"  Augmentation: {self.augment}")
#         if cache_size_limit:
#             print(f"  Cache size limit: {cache_size_limit} items")

#     def __len__(self) -> int:
#         return len(self.df)

#     def set_cache_mode(self, use_cache: bool):
#         """Enable or disable caching"""
#         self.use_cache = use_cache
#         print(f"NLSTDataset_catch caching: {'ENABLED' if use_cache else 'DISABLED'}")

#     def clear_cache(self):
#         """Clear all cached data"""
#         self.cache = [None] * len(self.df)
#         self.cache_stats = {
#             'hits': 0,
#             'misses': 0,
#             'size': 0,
#             'memory_usage': 0
#         }
#         print("NLSTDataset_catch cache cleared")

#     def get_cache_stats(self):
#         """Get current cache statistics"""
#         total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
#         hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
#         return {
#             'hits': self.cache_stats['hits'],
#             'misses': self.cache_stats['misses'],
#             'hit_rate': f"{hit_rate:.2f}%",
#             'cache_size': self.cache_stats['size'],
#             'memory_usage_mb': f"{self.cache_stats['memory_usage'] / (1024*1024):.2f} MB"
#         }

#     def _load_volume_data(self, idx: int):
#         """
#         Load and process volume data for a given index.
#         This is the expensive operation we want to cache.
#         """
#         row = self.df.iloc[idx]
#         nii_path = row.file_path
        
#         # Add error handling for corrupted NPZ files
#         try:
#             # Load NPZ file instead of NIfTI
#             npz_data = np.load(nii_path)
#             vol = npz_data['data'].astype(np.float32)
#             spacing = tuple(npz_data['spacing'])
#         except Exception as e:
#             print(f"Error loading file {nii_path}: {e}")
#             # Return a fallback sample with zeros
#             dummy_vol = np.zeros((448, 448, 3), dtype=np.float32)
#             chunks = torch.from_numpy(dummy_vol).permute(2, 0, 1).unsqueeze(0)
#             labels = row[self.labels].to_numpy(dtype=np.float32)
#             mask = (labels != -1)
#             spacing = (1.0, 1.0, 1.0)
#             return chunks, labels, mask, spacing
        
#         H, W, D = vol.shape
#         num_chunks = D // self.proc.chunk_depth

#         windows = []
#         for i in range(num_chunks):
#             arr = vol[:, :, i*self.proc.chunk_depth:(i+1)*self.proc.chunk_depth]
#             arr = np.clip(arr, self.proc.vmin, self.proc.vmax)
#             arr = (arr - self.proc.vmin) / (self.proc.vmax - self.proc.vmin)
#             arr = np.clip(arr, self.proc.eps, 1.0 - self.proc.eps)

#             t = torch.from_numpy(arr).permute(2, 0, 1)
#             t = F.interpolate(t.unsqueeze(0), size=self.proc.out_size,
#                               mode="bilinear", align_corners=False).squeeze(0)
#             if self.augment is not None:
#                 t = self.augment(t)
#             t = (t - 0.5) / 0.5
#             windows.append(t)

#         chunks = torch.stack(windows, dim=0)
#         labels = row[self.labels].to_numpy(dtype=np.float32)
#         mask = (labels != -1)
        
#         return chunks, labels, mask, spacing

#     def __getitem__(self, idx: int):
#         """
#         Get item with caching support.
#         """
#         # Check cache first
#         if self.use_cache and self.cache[idx] is not None:
#             self.cache_stats['hits'] += 1
#             return self.cache[idx]
        
#         # Cache miss - load data
#         self.cache_stats['misses'] += 1
        
#         # Load and process the data
#         chunks, labels, mask, spacing = self._load_volume_data(idx)
        
#         # Create sample
#         sample = (chunks, labels, mask, spacing)
        
#         # Cache the result if caching is enabled
#         if self.use_cache:
#             # Check cache size limit
#             if self.cache_size_limit and self.cache_stats['size'] >= self.cache_size_limit:
#                 # Simple LRU: clear first item
#                 self.cache[0] = None
#                 self.cache_stats['size'] -= 1
            
#             self.cache[idx] = sample
#             self.cache_stats['size'] += 1
            
#             # Estimate memory usage (rough calculation)
#             if isinstance(chunks, torch.Tensor):
#                 self.cache_stats['memory_usage'] += chunks.numel() * chunks.element_size()
        
#         return sample

#     def print_cache_stats(self):
#         """Print current cache statistics"""
#         stats = self.get_cache_stats()
#         print("\n=== NLSTDataset_catch Cache Statistics ===")
#         for key, value in stats.items():
#             print(f"  {key}: {value}")
#         print("=" * 40)


# Add this import at the top with other imports


# Add this focal loss class after the imports
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets, pos_weight=None):
        """
        Args:
            inputs: logits from model (N, C)
            targets: binary targets (N, C)
            pos_weight: positive class weights (optional)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss components
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply pos_weight if provided
        if pos_weight is not None:
            # For positive samples, multiply by pos_weight
            # For negative samples, keep weight as 1
            pos_weight_expanded = pos_weight.expand_as(inputs)
            focal_weight = torch.where(targets == 1, 
                                     focal_weight * pos_weight_expanded, 
                                     focal_weight)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight, reduction='none'
        )
        
        # Apply focal weighting
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# class UncertaintyWeightedSumNamed(nn.Module):
#     """
#     Pass a dict: {"cls": L_cls, "bbox": L_bbox, "depth": L_depth}
#     Optionally specify per-task 'use_half' flags.
#     """
#     def __init__(self, task_names: list[str],
#                  use_half_map: dict[str, bool] | None = None,
#                  clamp_logvar: tuple|None = (-10.0, 10.0)):
#         super().__init__()
#         self.names = list(task_names)
#         self.name_to_idx = {n:i for i,n in enumerate(self.names)}
#         self.use_half_map = use_half_map or {}
#         self.clamp_range = clamp_logvar
#         self.log_vars = nn.Parameter(torch.zeros(len(self.names)))

#     def forward(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
#         # Ensure we got all tasks
#         assert set(loss_dict.keys()) == set(self.names), \
#             f"Expected keys {self.names}, got {list(loss_dict.keys())}"

#         # Build aligned tensors
#         losses = []
#         halves = []
#         for n in self.names:
#             L = loss_dict[n]
#             L = L.mean() if L.ndim > 0 else L
#             losses.append(L)
#             halves.append(self.use_half_map.get(n, True))  # default True
#         L = torch.stack(losses)  # [T]

#         log_vars = self.log_vars
#         if self.clamp_range is not None:
#             low, high = self.clamp_range
#             log_vars = torch.clamp(log_vars, low, high)

#         precisions = torch.exp(-log_vars)  # [T]
#         weighted = precisions * L + log_vars  # [T]

#         # Apply 0.5 only to the tasks that want it
#         half_mask = torch.tensor(halves, dtype=weighted.dtype, device=weighted.device)
#         # half_mask is {True,False}; convert to {0.5,1.0}
#         factors = 0.5 * half_mask + 1.0 * (1 - half_mask)
#         weighted = factors * weighted
#         return weighted.sum()



class UncertaintyWeightedSumNamed(nn.Module):
    """
    Kendall–Gal multi-task weighting with:
      • per-task names
      • per-task 0.5 factor toggle (regression vs CE)
      • per-sample skipping (omit key OR set value=None/NaN)
      • optional global skip via use_half_map[name] == -1

    Forward accepts a dict: {name: loss_tensor or None}
    Only active tasks contribute; others are ignored (no penalty term).
    """
    def __init__(self,
                 task_names: list[str],
                 use_half_map: dict[str, int|bool] | None = None,
                 clamp_logvar: tuple | None = (-10.0, 10.0),
                 allow_missing: bool = True):
        super().__init__()
        self.names = list(task_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}
        self.use_half_map = use_half_map or {}
        self.clamp_range = clamp_logvar
        self.allow_missing = allow_missing

        # Track tasks that should *never* be used (global off)
        self.always_skip = {n for n, v in self.use_half_map.items() if v == -1}

        self.log_vars = nn.Parameter(torch.zeros(len(self.names)))

    def forward(self, loss_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        # Validate keys unless allow_missing
        if not self.allow_missing:
            assert set(loss_dict.keys()) == set(self.names), \
                f"Expected keys {self.names}, got {list(loss_dict.keys())}"

        active_names = []
        active_losses = []

        for n in self.names:
            if n in self.always_skip:
                continue  # globally disabled
            v = loss_dict.get(n, None)  # may be missing
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                # Treat NaN as "skip this sample"
                if torch.isnan(v).any():
                    continue
                v = v.mean() if v.ndim > 0 else v
                active_names.append(n)
                active_losses.append(v)

        if len(active_losses) == 0:
            # No active tasks this step
            return torch.zeros((), device=self.log_vars.device, dtype=torch.float32)

        L = torch.stack(active_losses)  # [A]
        device = L.device
        dtype = L.dtype

        # Select corresponding log_vars
        idx = torch.tensor([self.name_to_idx[n] for n in active_names],
                           device=device, dtype=torch.long)
        log_vars = self.log_vars.index_select(0, idx)  # [A]
        if self.clamp_range is not None:
            low, high = self.clamp_range
            log_vars = torch.clamp(log_vars, low, high)

        precisions = torch.exp(-log_vars)              # [A]
        weighted = precisions * L + log_vars           # [A]

        # Apply 0.5 only where desired (default True)
        halves = torch.tensor(
            [bool(self.use_half_map.get(n, True)) for n in active_names],
            device=device, dtype=torch.bool
        )
        factors = torch.where(halves,
                              torch.tensor(0.5, device=device, dtype=dtype),
                              torch.tensor(1.0, device=device, dtype=dtype))
        weighted = factors * weighted

        return weighted.sum()



import numpy as np
from sklearn.metrics import roc_auc_score

def _confusion_metrics_from_cm(cm: np.ndarray):
    """
    Given a KxK confusion matrix (rows=true, cols=pred),
    return per-class vectors and macro/weighted summaries.
    """
    cm = cm.astype(np.int64)
    K = cm.shape[0]
    total = cm.sum()
    row_sum = cm.sum(axis=1)  # support per true class
    col_sum = cm.sum(axis=0)

    tp = np.diag(cm).astype(float)
    fn = row_sum - tp
    fp = col_sum - tp
    tn = total - (tp + fn + fp)

    # safe divisions -> NaN when denominator==0
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = tp / (tp + fn)      # recall
        spec = tn / (tn + fp)
        prec = tp / (tp + fp)
        f1   = 2 * (prec * sens) / (prec + sens)

    # balanced accuracy (per-class)
    bal_acc = (sens + spec) / 2.0

    # macro: average over classes ignoring NaNs
    macro = lambda v: np.nanmean(v) if np.sum(~np.isnan(v)) > 0 else np.nan
    macro_sens = macro(sens)
    macro_spec = macro(spec)
    macro_prec = macro(prec)
    macro_f1   = macro(f1)
    macro_bal  = macro(bal_acc)

    # weighted by support (rows)
    w = row_sum.astype(float)
    w_sum = w.sum()
    def wavg(v):
        if w_sum == 0:
            return np.nan
        m = ~np.isnan(v)
        if m.sum() == 0:
            return np.nan
        return np.average(v[m], weights=w[m])

    weighted_sens = wavg(sens)
    weighted_spec = wavg(spec)
    weighted_prec = wavg(prec)
    weighted_f1   = wavg(f1)
    weighted_bal  = wavg(bal_acc)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sens": sens, "spec": spec, "prec": prec, "f1": f1, "bal_acc": bal_acc,
        "support": row_sum.astype(int),
        "macro": {
            "sens": macro_sens, "spec": macro_spec, "prec": macro_prec, "f1": macro_f1, "bal_acc": macro_bal,
        },
        "weighted": {
            "sens": weighted_sens, "spec": weighted_spec, "prec": weighted_prec, "f1": weighted_f1, "bal_acc": weighted_bal,
        }
    }

def _exactly_one_hot_valid(tgt_slice: np.ndarray) -> bool:
    """Valid if no -1 present and exactly one '1' (others 0)."""
    if np.any(tgt_slice < 0):  # contains -1
        return False
    # allow float fuzz
    return np.isclose(tgt_slice.sum(), 1.0)

def _print_group_table(group_name: str, class_names: list[str], stats: dict):
    K = len(class_names)
    print(f"\n[{group_name}] per-class 1-vs-all metrics")
    print(f"{'class':20s} {'supp':>5s}  {'sens':>6s} {'spec':>6s} {'prec':>6s} {'f1':>6s} {'balAcc':>7s}")
    
    # Log the same information
    import logging
    logging.info(f"[{group_name}] per-class 1-vs-all metrics")
    logging.info(f"{'class':20s} {'supp':>5s}  {'sens':>6s} {'spec':>6s} {'prec':>6s} {'f1':>6s} {'balAcc':>7s}")
    
    for i in range(K):
        c = class_names[i]
        supp = stats['support'][i]
        sens = stats['sens'][i]
        spec = stats['spec'][i]
        prec = stats['prec'][i]
        f1   = stats['f1'][i]
        ba   = stats['bal_acc'][i]
        fmt = lambda v: f"{v:.3f}" if np.isfinite(v) else "  nan"
        line = f"{c:20s} {supp:5d}  {fmt(sens):>6s} {fmt(spec):>6s} {fmt(prec):>6s} {fmt(f1):>6s} {fmt(ba):>7s}"
        print(line)
        logging.info(line)

    m = stats['macro']; w = stats['weighted']
    def fmt(v): return f"{v:.3f}" if np.isfinite(v) else "nan"
    macro_line = f"macro avg:     sens={fmt(m['sens'])}  spec={fmt(m['spec'])}  prec={fmt(m['prec'])}  f1={fmt(m['f1'])}  balAcc={fmt(m['bal_acc'])}"
    weighted_line = f"weighted avg:  sens={fmt(w['sens'])}  spec={fmt(w['spec'])}  prec={fmt(w['prec'])}  f1={fmt(w['f1'])}  balAcc={fmt(w['bal_acc'])}"
    
    print(macro_line)
    print(weighted_line)
    logging.info(macro_line)
    logging.info(weighted_line)
