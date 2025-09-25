# =======================
# Multitask Loss Module
# =======================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class UncertaintyWeightedSumNamed(nn.Module):
    """
    Kendall–Gal multi-task uncertainty weighting with task names.

    forward(loss_dict) where loss_dict maps name->tensor (scalar) or None.
    Skips missing tasks; each active task i uses:   0.5 * (exp(-s_i)*L_i + s_i)
    If you don't want the 0.5 factor for classification, set use_half_map[name]=False.
    """
    def __init__(
        self,
        task_names: List[str],
        use_half_map: Optional[Dict[str, bool]] = None,
        clamp_logvar: Optional[Tuple[float, float]] = (-10.0, 10.0),
    ):
        super().__init__()
        self.names = list(task_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}
        self.use_half_map = use_half_map or {n: False for n in self.names}  # default: no 0.5 factor for CE/BCE
        self.clamp_range = clamp_logvar
        self.log_vars = nn.Parameter(torch.zeros(len(self.names)))

    def forward(self, loss_dict: Dict[str, Optional[Tensor]]) -> Tensor:
        active_names, active_losses = [], []
        for n in self.names:
            v = loss_dict.get(n, None)
            if v is None:
                continue
            if torch.isnan(v).any():
                continue
            v = v.mean() if v.ndim > 0 else v
            active_names.append(n)
            active_losses.append(v)

        if len(active_losses) == 0:
            # no active tasks -> return zero (no grad)
            return torch.zeros((), device=self.log_vars.device, dtype=self.log_vars.dtype)

        L = torch.stack(active_losses)  # [A]
        idx = torch.tensor([self.name_to_idx[n] for n in active_names],
                           device=L.device, dtype=torch.long)
        log_vars = self.log_vars.index_select(0, idx)
        if self.clamp_range is not None:
            low, high = self.clamp_range
            log_vars = torch.clamp(log_vars, low, high)

        precisions = torch.exp(-log_vars)
        weighted = precisions * L + log_vars

        # Optional 0.5 factor per task (commonly True for regression, False for CE/BCE)
        halves = torch.tensor([bool(self.use_half_map.get(n, False)) for n in active_names],
                              device=L.device, dtype=torch.bool)
        factors = torch.where(halves,
                              torch.tensor(0.5, device=L.device, dtype=L.dtype),
                              torch.tensor(1.0, device=L.device, dtype=L.dtype))
        return (factors * weighted).sum()

    # Convenience: peek at current s (=log_var) and precisions for debugging
    def snapshot(self) -> Dict[str, float]:
        out = {}
        with torch.no_grad():
            for n, i in self.name_to_idx.items():
                out[n] = float(self.log_vars[i].item())
        return out

    def precisions(self) -> Dict[str, float]:
        out = {}
        with torch.no_grad():
            for n, i in self.name_to_idx.items():
                out[n] = float(torch.exp(-self.log_vars[i]).item())
        return out


@dataclass
class MultiTaskLossConfig:
    label_cols: List[str]                         # global order of heads (T)
    groups: List[List[str]]                       # list of mutually-exclusive groups (each is a list of label names)
    binary_names: List[str]                       # independent binary heads
    ce_class_weights: Optional[Dict[str, Tensor]] = None  # group_name -> [C] weights (on SAME device as logits)
    pos_weight_full: Optional[Tensor] = None      # [T] aligned to label_cols (used only for binary heads)
    group_names: Optional[List[str]] = None       # optional names for groups; otherwise auto "group_1", "group_2", ...
    use_uncertainty: bool = True                  # enable Kendall–Gal wrapper
    clamp_logvar: Tuple[float, float] = (-10.0, 10.0)
    accum_steps: int = 1                          # divide final loss for grad accumulation


class MultiTaskLoss(nn.Module):
    """
    End-to-end loss for:
      • Mutually exclusive groups (CE with softmax over group)
      • Independent binaries (BCE with logits)
      • Missing labels (-1) skipped automatically
      • Optional per-group CE class weights and per-label BCE pos_weight
      • Optional Kendall–Gal uncertainty weighting across (group-level + binary) tasks

    Inputs (per forward):
      logits: [B, T] or [T]  (T == len(label_cols))
      targets: [B, T] or [T] (0/1 for known, -1 for missing)
      mask: optional [B, T] or [T] boolean (extra skip switch)

    Returns:
      total_loss: scalar tensor
      info: dict with rich breakdown (per-task losses, counts, snapshot of log_vars, etc.)
    """
    def __init__(self, cfg: MultiTaskLossConfig):
        super().__init__()
        self.cfg = cfg

        # ---- names & indices
        lbls = cfg.label_cols
        self.name2idx = {n: i for i, n in enumerate(lbls)}

        # group naming
        if cfg.group_names is not None:
            assert len(cfg.group_names) == len(cfg.groups), "group_names must match groups length"
            self.group_names = list(cfg.group_names)
        else:
            self.group_names = [f"group_{i+1}" for i in range(len(cfg.groups))]

        # build index groups (validate presence)
        self.idx_groups: Dict[str, List[int]] = {}
        for g_name, g_list in zip(self.group_names, cfg.groups):
            idxs = []
            for n in g_list:
                if n not in self.name2idx:
                    raise ValueError(f"Group label '{n}' not found in label_cols.")
                idxs.append(self.name2idx[n])
            self.idx_groups[g_name] = idxs

        # binaries
        self.binary_idxs: List[int] = []
        self.binary_names: List[str] = []
        for n in cfg.binary_names:
            if n not in self.name2idx:
                raise ValueError(f"Binary label '{n}' not found in label_cols.")
            self.binary_idxs.append(self.name2idx[n])
            self.binary_names.append(n)

        # ---- uncertainty wrapper over (groups + binaries)
        if cfg.use_uncertainty:
            all_task_names = self.group_names + self.binary_names
            use_half_map = {n: False for n in all_task_names}  # CE/BCE → no 0.5
            self.wrapper = UncertaintyWeightedSumNamed(
                task_names=all_task_names,
                use_half_map=use_half_map,
                clamp_logvar=cfg.clamp_logvar,
            )
        else:
            self.wrapper = None

        # store CE weights dict (can be None → uniform)
        # Expect: group_name -> [C] tensor (device set by caller)
        self.ce_w = cfg.ce_class_weights or {}

        # pos_weight across ALL labels (aligned to label_cols), but used only for binaries
        # Expect: [T] on same device as logits during forward (we'll .to(logits.device) on use)
        self.pos_weight_full = cfg.pos_weight_full

        # buffers purely for type consistency (not parameters)
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    # ---------- helpers ----------
    @staticmethod
    def _ensure_batched(x: Tensor, T: int) -> Tensor:
        # Accept [T] or [B,T] -> return [B,T]
        if x.dim() == 1:
            assert x.shape[0] == T, f"Expected T={T}, got {x.shape}"
            return x.unsqueeze(0)
        assert x.dim() == 2 and x.shape[1] == T, f"Expected [B,T] with T={T}, got {tuple(x.shape)}"
        return x

    # ---------- forward ----------
    def forward(
        self,
        logits: Tensor,     # [B,T] or [T]
        targets: Tensor,    # [B,T] or [T], entries in {0,1,-1}
        mask: Optional[Tensor] = None,  # [B,T] or [T] boolean (optional)
    ) -> Tuple[Tensor, Dict[str, Union[float, Dict, List]]]:

        device = logits.device
        T = logits.shape[-1]
        logits = self._ensure_batched(logits, T).float()
        targets = self._ensure_batched(targets.to(device=device, dtype=torch.float32), T)

        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            mask = self._ensure_batched(mask.to(device=device, dtype=torch.bool), T)

        B = logits.size(0)

        # Optional: pos_weight vector for binaries
        pos_weight = None
        if self.pos_weight_full is not None:
            pos_weight = self.pos_weight_full.to(device=device, dtype=torch.float32)

        total = logits.new_tensor(0.0)
        per_sample_breakdown = []

        # Process each sample independently (simpler logic & matching common training loops with BS=1)
        for b in range(B):
            l_row = logits[b]   # [T]
            t_row = targets[b]  # [T]
            m_row = mask[b]     # [T]

            loss_dict: Dict[str, Optional[Tensor]] = {}
            dbg = {"groups": {}, "binaries": {}}

            # --- groups: CE over each group's slice ---
            for g_name, idxs in self.idx_groups.items():
                # Skip group if ANY label missing (-1)
                if any((t_row[i].item() == -1) for i in idxs):
                    loss_dict[g_name] = None
                    dbg["groups"][g_name] = {"skipped": True}
                    continue

                # target class by argmax of one-hot slice (robust to double-1 noise)
                tgt_slice = t_row[idxs]                 # [C]
                tgt_class = torch.argmax(tgt_slice).long()  # scalar
                lg = l_row[idxs]                        # [C]
                w = self.ce_w.get(g_name, None)         # [C] or None
                if w is not None:
                    w = w.to(device=lg.device, dtype=lg.dtype)

                ce = F.cross_entropy(lg.unsqueeze(0), tgt_class.unsqueeze(0), weight=w)
                loss_dict[g_name] = ce
                dbg["groups"][g_name] = {"skipped": False, "loss": float(ce.detach().item()), "target_class": int(tgt_class)}

            # --- binaries: BCE per independent head ---
            for j, name in zip(self.binary_idxs, self.binary_names):
                if (m_row[j]) and (t_row[j].item() != -1):
                    pw = None
                    if pos_weight is not None:
                        pw = pos_weight[j:j+1]
                    bce = F.binary_cross_entropy_with_logits(
                        l_row[j:j+1], t_row[j:j+1], pos_weight=pw, reduction='mean'
                    )
                    loss_dict[name] = bce
                    dbg["binaries"][name] = {"skipped": False, "loss": float(bce.detach().item()), "target": int(t_row[j].item())}
                else:
                    loss_dict[name] = None
                    dbg["binaries"][name] = {"skipped": True}

            # --- combine ---
            if self.wrapper is not None:
                combined = self.wrapper(loss_dict)
            else:
                # mean of available losses (if none active, 0.0)
                vals = [v for v in loss_dict.values() if v is not None]
                combined = (torch.stack(vals).mean() if len(vals) > 0
                            else logits.new_tensor(0.0))

            per_sample_breakdown.append({"loss_dict": {k: (float(v.detach().item()) if v is not None else None)
                                                        for k, v in loss_dict.items()},
                                         "detail": dbg})
            total = total + combined

        # Average over batch and scale by accum_steps
        total = total / max(B, 1)
        total = total / float(max(1, self.cfg.accum_steps))

        info = {
            "batch_size": B,
            "per_sample": per_sample_breakdown,
            "used_uncertainty": self.wrapper is not None,
        }
        if self.wrapper is not None:
            # include current log_vars and precisions snapshot for easy introspection
            info["log_vars"] = self.wrapper.snapshot()
            info["precisions"] = self.wrapper.precisions()

        return total, info


# -------------------------
# (Optional) Helper: build CE weights from a Pandas DataFrame
# Each group's class weight = normalized inverse frequency on the train split.
# -------------------------
def build_ce_class_weights_from_df(
    df, 
    label_cols: List[str],
    groups: List[List[str]],
    group_names: Optional[List[str]] = None,
    split_col: str = "split",
    split_value: str = "train",
    eps: float = 1e-6,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Tensor]:
    """
    Returns {group_name: tensor([C])} suitable for CE 'weight' argument.
    Assumes each column in group is 0/1 with -1 for missing.
    """
    import numpy as np
    if group_names is None:
        group_names = [f"group_{i+1}" for i in range(len(groups))]
    df = df[df[split_col] == split_value]
    out = {}
    for g_name, cols in zip(group_names, groups):
        sub = df[cols].replace(-1, np.nan)
        counts = []
        for c in cols:
            # count of positives for class c
            counts.append(np.nansum(sub[c].values == 1))
        counts = np.array(counts, dtype=np.float32) + eps
        inv = 1.0 / counts
        weights = (inv / inv.sum()) * len(counts)  # normalized inverse freq
        t = torch.tensor(weights, device=device if device is not None else "cpu", dtype=torch.float32)
        out[g_name] = t
    return out




import torch

# ----- Define label space -----
label_cols = [
    # histology (3)
    'is_scc','is_adenocarcinoma','is_other',
    # metastasis (4)
    'is_M0','is_M1A','is_M1B','is_M1C',
    # node (4)
    'is_N0','is_N1','is_N2','is_N3',
    # tumor (5)
    'is_T0','is_T1','is_T2','is_T3','is_T4',
    # independent binary
    'is_mw'
]
groups = [
    ['is_scc','is_adenocarcinoma','is_other'],
    ['is_M0','is_M1A','is_M1B','is_M1C'],
    ['is_N0','is_N1','is_N2','is_N3'],
    ['is_T0','is_T1','is_T2','is_T3','is_T4'],
]
group_names = ['histology','metastasis','node','tumor']
binary_names = ['is_mw']

T = len(label_cols)

# ----- Fake one sample -----
# Logits (unnormalized): shape [T]
torch.manual_seed(0)
logits = torch.randn(T)

# Targets: one-hot inside each group, 0/1 for binaries, -1 for missing
targets = torch.full((T,), -1.0)  # start all missing
# Make valid groups with a single 1 each:
targets[0:3]  = torch.tensor([0,1,0])    # histology -> class 1
targets[3:7]  = torch.tensor([1,0,0,0])  # metastasis -> class 0
targets[7:11] = torch.tensor([0,0,1,0])  # node -> class 2
targets[11:16]= torch.tensor([0,1,0,0,0])# tumor -> class 1
# Binary label (known)
targets[16]   = 1.0

# Mask: all heads considered (True). You can turn any off if needed.
mask = torch.ones(T, dtype=torch.bool)

# ----- Optional weights -----
# pos_weight across all labels (used only for binaries). Here: put 5.0 at 'is_mw', 1.0 elsewhere.
pos_weight_full = torch.ones(T)
pos_weight_full[label_cols.index('is_mw')] = 5.0

# Example CE class weights per-group (use None to keep uniform)
ce_w = {
    'histology': torch.tensor([1.0, 2.0, 1.0]),       # weights per class
    'metastasis': torch.tensor([1.0, 1.0, 1.0, 1.0]),
    'node': torch.tensor([1.0, 1.0, 1.0, 1.0]),
    'tumor': torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
}

# ----- Build loss -----
cfg = MultiTaskLossConfig(
    label_cols=label_cols,
    groups=groups,
    binary_names=binary_names,
    ce_class_weights=ce_w,         # or None
    pos_weight_full=pos_weight_full,
    group_names=group_names,
    use_uncertainty=True,          # Kendall–Gal on {histology, metastasis, node, tumor, is_mw}
    clamp_logvar=(-10, 10),
    accum_steps=1,
)
mtl = MultiTaskLoss(cfg)

# ----- Compute -----
total_loss, info = mtl(logits, targets, mask)
print("Total loss:", float(total_loss.item()))
print("Per-task losses:", info["per_sample"][0]["loss_dict"])
print("log_vars:", info.get("log_vars", {}))
print("precisions:", info.get("precisions", {}))

# Backprop just to confirm it’s differentiable:
total_loss.backward()
print("Grad on log_vars (if uncertainty enabled):",
      None if not cfg.use_uncertainty else mtl.wrapper.log_vars.grad)
