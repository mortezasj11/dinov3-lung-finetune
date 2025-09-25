# Histology Fine-Tuning (Single GPU)

This repository contains a single-GPU training pipeline for multi-task histology prediction with DINOv2 features, transformer aggregation, confusion-matrix based evaluation, optional focal loss, cosine LR with warmup, and robust logging.

The pipeline can run in two deployment modes with different storage layouts:
- MNT: data staged under /shared by a mount/copy (mnt-job-submit.yaml)
- SCC: similar flow but different launcher and job spec (scc-job-submit.yaml)

Below is a quick start guide for both, plus details on arguments, outputs, and troubleshooting.

## Contents
- Code entrypoint: `2_run_fineTune_single_gpu_learnable_loss-wrapper.py`
- Utilities and model: `model_utils_single_gpu.py`
- Launchers:
  - MNT: `run_fineTune_single_gpu_launcher_mnt.py`
  - SCC: `run_fineTune_single_gpu_launcher_scc.py`
- K8s Jobs:
  - `mnt-job-submit.yaml`
  - `scc-job-submit.yaml`
- Logs: `logs/*.log`, per-run `output/.../training.log`

## Data Layouts (MNT vs SCC)

The code autodetects data copied to /shared. It supports both raw NIfTI-derived NPZ trees and “_mnt” variants.

- Expected original roots:
  - `/rsrch7/.../FineTune/ready_finetune_max192_npz`
  - `/rsrch7/.../FineTune/ready_finetune_max192`

- Typical init copy in MNT job:
  - `cp -r .../ready_finetune_max192_npz/ /shared/`
  - or `cp -r .../ready_finetune_max192/ /shared/`

The loader will map file paths in the CSV to /shared/ready_finetune_max192_npz (or /shared/ready_finetune_max192) and switch `.nii.gz → .npz` as needed. If /shared/ready_finetune_max192_npz[_mnt] exists, it will be preferred.

## Running: MNT

1) Stage data via `mnt-job-submit.yaml` (init container copies to /shared). Verify `/shared/ready_finetune_max192_npz` or `/shared/ready_finetune_max192` exists in pod.

2) Use the MNT launcher locally or as container entrypoint:

```bash
python run_fineTune_single_gpu_launcher_mnt.py \
  --csv /rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/CSVs/ready_finetune_t1919_v500_all_info.csv \
  --groups "[['is_scc','is_adenocarcinoma','is_other'],['is_M0','is_M1A','is_M1B','is_M1C'],['is_N0','is_N1','is_N2','is_N3'],['is_T0','is_T1','is_T2','is_T3','is_T4']]" \
  --binary-names is_mw \
  --output /rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/output/outputSep20_MNT/output18_ALL_ME_waqas_tau1.0_unbalanced_weighted_lr0.001 \
  --lr 1e-3 --epochs 50 --accum-steps 200 --val-every 3000 --print-every 600
```

Example with custom group names (main script supports `--group-names` for display):

```bash
python run_fineTune_single_gpu_launcher_mnt.py \
  --csv /rsrch7/.../FineTune/CSVs/ready_finetune_t1919_v500_all_info.csv \
  --groups "[['is_scc','is_adenocarcinoma','is_other'],['is_M0','is_M1A','is_M1B','is_M1C'],['is_N0','is_N1','is_N2','is_N3'],['is_T0','is_T1','is_T2','is_T3','is_T4']]" \
  --group-names histology metastasis node tumor \
  --binary-names is_mw \
  --output /rsrch7/.../outputSep20_MNT/output18_ALL_ME_example \
  --lr 1e-3 --epochs 50 --accum-steps 200 --val-every 3000 --print-every 600
```

3) Check logs in `{output}/training.log` and pod log. Validation tables and confusion matrices are printed and logged.

## Running: SCC

1) Use `scc-job-submit.yaml` with the SCC image and pathing.

2) Launch locally for debugging:

```bash
python run_fineTune_single_gpu_launcher_scc.py \
  --csv /rsrch7/.../FineTune/CSVs/ready_finetune_tXXXX_vYYY.csv \
  --groups "[[...],[...],[...],[...]]" \
  --binary-names is_mw \
  --output /rsrch7/.../FineTune/output/outputSepXX/OUTPUT_DIR
```

## Key Arguments (common)

- `--csv`: dataset CSV; must include split column and label columns.
- `--groups`: Python-list string of mutually exclusive groups (each sums to 1).
- `--binary-names`: independent binary tasks (0/1, -1 for missing) (can be empty).
- `--group-names` (main script): optional display names for each group.
- `--accum-steps`: gradient accumulation; ensure `accum-steps ≤ batches_per_epoch` or the code will “flush” at epoch end.
- `--lr`: main head LR; base LR = 0.2× (in default optimizer grouping).
- `--optimizer`: adamw|adam|sgd (AdamW recommended).
- `--lse-tau`: temperature for LSE pooling (if not waqas attention).
- `--waqas-way`: enable attention pooling.
- `--use-focal-loss`: focal loss on group heads instead of CE.
- `--balanced-sampling`/`--class-weights`: sampling and loss weighting.
- `--warmup-steps`: manual linear warmup integrated with cosine restarts.
- `--print-every`, `--val-every`: logging cadence (iterations).
- `--max-chunks`: maximum chunks per sample (validation keeps all; training keeps a random fraction r∈[0.3,1.0]).

## Training Details

- Label smoothing for CE (group heads): 0.05 (configurable in code).
- Focal loss available via `--use-focal-loss` (CE path otherwise).
- Optimizer: per-group weight decay applied; AdamW recommended.
- Scheduler: CosineAnnealingWarmRestarts + manual warmup; warmup scales per param-group `initial_lr` and respects accumulation.
- Accumulation flush: If an epoch ends mid-accum, we perform one optimizer step to avoid “no updates this epoch”.

## Outputs

- `{output}/training.log`: all prints mirrored to logging, plus validation tables and confusion matrices.
- `{output}/checkpoints/model_ep{E}_it{S}.pt` and split parts (base/aggregator) at save intervals.
- Metrics stream in `metrics` directory (from MetricsLogger).

## Troubleshooting

- No training logs in training.log:
  - `print_every` may be large relative to steps; lower it.
  - Another logger may have been configured; we set `force=True` to ensure FileHandler is active.

- “First real step → LR” appears late:
  - Ensure `accum-steps ≤ batches/epoch`, or rely on the end-of-epoch flush (added).

- Data not found under /shared:
  - Verify initContainer copy in YAML.
  - The loader maps original roots to `/shared/ready_finetune_max192_npz[_mnt]` or `/shared/ready_finetune_max192`.

- Overfitting:
  - Increase dropout, weight decay, and augmentations.
  - Use label smoothing or focal loss.
  - Consider longer, gentler cosine schedule.

## Example: Minimal Direct Call

```bash
python 2_run_fineTune_single_gpu_learnable_loss-wrapper.py \
  --csv /rsrch7/.../FineTune/CSVs/ready_finetune.csv \
  --groups "[['is_scc','is_adenocarcinoma','is_other'],['is_M0','is_M1A','is_M1B','is_M1C'],['is_N0','is_N1','is_N2','is_N3'],['is_T0','is_T1','is_T2','is_T3','is_T4']]" \
  --binary-names is_mw \
  --output ./output_single_gpu_learnable \
  --epochs 5 --lr 1e-3 --accum-steps 100 --print-every 50 --val-every 200
```

## Notes
- Validation uses all chunks (no drop), with confusion matrices and macro/weighted metrics per group.
- Training randomly keeps a fraction of chunks per sample (r∈[0.3,1.0]) and applies simple order augment.
- Binary tasks (if any) use BCE with optional `pos_weight`.

## Hyperparameter Grid

| Run | Learning Rate | LSE Tau | Waqas Attention | Focal Loss | Job Name Hint |
| --- | ------------- | ------- | --------------- | ---------- | ------------- |
| 1   | 2e-4          | 0.33    | True            | True       | lr2e-4_tau0.33_waqasT_focalT |
| 2   | 2e-4          | 0.33    | True            | False      | lr2e-4_tau0.33_waqasT_focalF |
| 3   | 2e-4          | 0.33    | False           | True       | lr2e-4_tau0.33_waqasF_focalT |
| 4   | 2e-4          | 0.33    | False           | False      | lr2e-4_tau0.33_waqasF_focalF |
| 5   | 2e-4          | 1.0     | True            | True       | lr2e-4_tau1.0_waqasT_focalT |
| 6   | 2e-4          | 1.0     | True            | False      | lr2e-4_tau1.0_waqasT_focalF |
| 7   | 2e-4          | 1.0     | False           | True       | lr2e-4_tau1.0_waqasF_focalT |
| 8   | 2e-4          | 1.0     | False           | False      | lr2e-4_tau1.0_waqasF_focalF |
| 9   | 1e-3          | 0.33    | True            | True       | lr1e-3_tau0.33_waqasT_focalT |
| 10  | 1e-3          | 0.33    | True            | False      | lr1e-3_tau0.33_waqasT_focalF |
| 11  | 1e-3          | 0.33    | False           | True       | lr1e-3_tau0.33_waqasF_focalT |
| 12  | 1e-3          | 0.33    | False           | False      | lr1e-3_tau0.33_waqasF_focalF |
| 13  | 1e-3          | 1.0     | True            | True       | lr1e-3_tau1.0_waqasT_focalT |
| 14  | 1e-3          | 1.0     | True            | False      | lr1e-3_tau1.0_waqasT_focalF |
| 15  | 1e-3          | 1.0     | False           | True       | lr1e-3_tau1.0_waqasF_focalT |
| 16  | 1e-3          | 1.0     | False           | False      | lr1e-3_tau1.0_waqasF_focalF |
| 17  | 5e-3          | 0.33    | True            | True       | lr5e-3_tau0.33_waqasT_focalT |
| 18  | 5e-3          | 0.33    | True            | False      | lr5e-3_tau0.33_waqasT_focalF |
| 19  | 5e-3          | 0.33    | False           | True       | lr5e-3_tau0.33_waqasF_focalT |
| 20  | 5e-3          | 0.33    | False           | False      | lr5e-3_tau0.33_waqasF_focalF |
| 21  | 5e-3          | 1.0     | True            | True       | lr5e-3_tau1.0_waqasT_focalT |
| 22  | 5e-3          | 1.0     | True            | False      | lr5e-3_tau1.0_waqasT_focalF |
| 23  | 5e-3          | 1.0     | False           | True       | lr5e-3_tau1.0_waqasF_focalT |
| 24  | 5e-3          | 1.0     | False           | False      | lr5e-3_tau1.0_waqasF_focalF |

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




