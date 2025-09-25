# Histology Fine-Tuning (Single GPU)

This repository contains a single-GPU training pipeline for multi-task histology prediction using **pre-trained DINOv2 features from 3 slices of CT scans**. The pipeline employs a **Multiple Instance Learning (MIL) approach** to fine-tune a transformer-based model for histology classification tasks on whole CT, optimized for **A100 GPU** deployment.

## Key Features
- **DINOv2 Integration**: Uses pre-trained DINOv2 features from 3 slices of CT scans
- **Multi-task Learning**: Handles mutually exclusive groups (histology, metastasis, node, tumor) + binary tasks
- **Transformer Aggregation**: Attention-based or LSE pooling for feature aggregation
- **Advanced Loss Functions**: Cross-entropy with label smoothing, optional focal loss
- **Uncertainty-Aware Task Balancing** (extended Kendall–Gal):
  - Per-task uncertainty scaling
  - Safely skips missing targets (returns zero loss if no tasks are active)
  - Supports regression/classification scaling

- **Smart Training**: Cosine LR with warmup, gradient accumulation, balanced sampling
- **Robust Evaluation**: Confusion matrices, macro/weighted metrics per group
- **Flexible Deployment**: MNT and SCC Kubernetes job modes
- **Comprehensive Logging**: Training logs, metrics tracking, checkpoint management

## Deployment Modes
- **MNT**: Data staged under /shared by mount/copy (`mnt-job-submit.yaml`)
- **SCC**: Similar flow with different launcher and job spec (`scc-job-submit.yaml`)

## Quick Start

**Files**: Main script `2_run_fineTune_single_gpu_learnable_loss-wrapper.py`, launchers for MNT/SCC, K8s job YAMLs

**Data**: Code autodetects data under `/shared/` (copied from `/rsrch7/.../ready_finetune_max192_npz`). Supports both NPZ and NIfTI formats with automatic `.nii.gz → .npz` conversion.

## Usage

### MNT Mode
```bash
# 1. Stage data via mnt-job-submit.yaml
# 2. Run training
python run_fineTune_single_gpu_launcher_mnt.py \
  --csv /path/to/ready_finetune_t1919_v500_all_info.csv \
  --groups "[['is_scc','is_adenocarcinoma','is_other'],['is_M0','is_M1A','is_M1B','is_M1C'],['is_N0','is_N1','is_N2','is_N3'],['is_T0','is_T1','is_T2','is_T3','is_T4']]" \
  --binary-names is_mw \
  --output /path/to/output \
  --lr 1e-3 --epochs 50 --accum-steps 200
```

### SCC Mode
```bash
python run_fineTune_single_gpu_launcher_scc.py \
  --csv /path/to/ready_finetune_tXXXX_vYYY.csv \
  --groups "[[...],[...],[...],[...]]" \
  --binary-names is_mw \
  --output /path/to/output
```

## Key Arguments

- `--csv`: Dataset CSV with split and label columns
- `--groups`: Mutually exclusive groups (e.g., histology, metastasis, node, tumor)
- `--binary-names`: Independent binary tasks (0/1, -1 for missing)
- `--lr`: Learning rate (base LR = 0.2× main head LR)
- `--waqas-way`: Enable attention pooling (vs LSE pooling)
- `--use-focal-loss`: Use focal loss instead of cross-entropy
- `--balanced-sampling`: Enable balanced sampling
- `--accum-steps`: Gradient accumulation steps

## Training Details

- **Loss**: Cross-entropy with label smoothing (0.05) or focal loss
- **Optimizer**: AdamW with per-group weight decay
- **Scheduler**: CosineAnnealingWarmRestarts + linear warmup
- **Sampling**: Random chunk fraction (0.3-1.0) for training, all chunks for validation
- **Outputs**: `{output}/training.log`, checkpoints, metrics directory

## Hyperparameter Grid Example (which based on this I change the lines in launcher as)

**Note**: Update the launcher file with:
```python
DEFAULT_LEARNING_RATE = 2e-4        # Learning rate for training
DEFAULT_LSE_TAU = 1.0                # Temperature parameter for LSE aggregation
DEFAULT_WAQAS_WAY = True             # Set to True to use Waqas attention aggregation
FOCAL_LOSS = False                  # Set to True to use focal loss
OUTPUT_DIR = f"output6_ALL_ME_{get_output_dir_name()}"
```

| Run | Learning Rate | LSE Tau | Waqas Attention | Focal Loss | Job Name Hint |
| --- | ------------- | ------- | --------------- | ---------- | ------------- |
| 1   | 2e-4          | 0.33    | True            | True       | lr2e-4_tau0.33_waqasT_focalT |
| 2   | 2e-4          | 0.33    | True            | False      | lr2e-4_tau0.33_waqasT_focalF |

| 5   | 2e-4          | 1.0     | True            | True       | lr2e-4_tau1.0_waqasT_focalT |
| 6   | 2e-4          | 1.0     | True            | False      | lr2e-4_tau1.0_waqasT_focalF |

| 9   | 1e-3          | 0.33    | True            | True       | lr1e-3_tau0.33_waqasT_focalT |
| 10  | 1e-3          | 0.33    | True            | False      | lr1e-3_tau0.33_waqasT_focalF |

| 13  | 1e-3          | 1.0     | True            | True       | lr1e-3_tau1.0_waqasT_focalT |
| 14  | 1e-3          | 1.0     | True            | False      | lr1e-3_tau1.0_waqasT_focalF |

| 17  | 5e-3          | 0.33    | True            | True       | lr5e-3_tau0.33_waqasT_focalT |
| 18  | 5e-3          | 0.33    | True            | False      | lr5e-3_tau0.33_waqasT_focalF |

| 21  | 5e-3          | 1.0     | True            | True       | lr5e-3_tau1.0_waqasT_focalT |
| 22  | 5e-3          | 1.0     | True            | False      | lr5e-3_tau1.0_waqasT_focalF |


## Kubernetes Deployment

### MNT Job Submission
```bash
# Set up kubectl access
export PATH="/rsrch1/ip/msalehjahromi/.kube:$PATH"
mkdir -p ~/.kube
ln -sf /rsrch1/ip/msalehjahromi/.kube/config ~/.kube/config

# Submit MNT job
job-runner.sh mnt-job-submit.yaml
```

### SCC Job Submission
```bash
# Submit SCC job
job-runner.sh scc-job-submit.yaml

# Or apply directly
kubectl apply -f scc-job-submit.yaml
```

### Job Management
```bash
# Delete specific job
kubectl delete job -n yn-gpu-workload msalehjahromi-histology6

# Monitor system resources
iostat -xz 1 5
```




