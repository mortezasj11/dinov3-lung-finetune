import subprocess
import os, sys, inspect
import logging
import argparse
import pandas as pd
import yaml
from pathlib import Path
import socket
import torch


# | Run | DEFAULT_LEARNING_RATE | DEFAULT_LSE_TAU | DEFAULT_WAQAS_WAY | DEFAULT_USE_FOCAL_LOSS | k8s (job name hint)                  |
# | --- | --------------------- | --------------- | ----------------- | ---------------------- | ------------------------------------ |
# | 1   | 2e-4                  | 0.33            | True              | True                   | lr2e-4_tau0.33_waqasT_focalT         |
# | 2   | 2e-4                  | 0.33            | True              | False                  | lr2e-4_tau0.33_waqasT_focalF         |
# | 3   | 2e-4                  | 0.33            | False             | True                   | lr2e-4_tau0.33_waqasF_focalT         |
# | 4   | 2e-4                  | 0.33            | False             | False                  | lr2e-4_tau0.33_waqasF_focalF         |
# | 5   | 2e-4                  | 1.0             | True              | True                   | lr2e-4_tau1.0_waqasT_focalT          |
# | 6   | 2e-4                  | 1.0             | True              | False                  | lr2e-4_tau1.0_waqasT_focalF          |
# | 7   | 2e-4                  | 1.0             | False             | True                   | lr2e-4_tau1.0_waqasF_focalT          |
# | 8   | 2e-4                  | 1.0             | False             | False                  | lr2e-4_tau1.0_waqasF_focalF          |
# | 9   | 1e-3                  | 0.33            | True              | True                   | lr1e-3_tau0.33_waqasT_focalT         |
# | 10  | 1e-3                  | 0.33            | True              | False                  | lr1e-3_tau0.33_waqasT_focalF         |
# | 11  | 1e-3                  | 0.33            | False             | True                   | lr1e-3_tau0.33_waqasF_focalT         |
# | 12  | 1e-3                  | 0.33            | False             | False                  | lr1e-3_tau0.33_waqasF_focalF         |
# | 13  | 1e-3                  | 1.0             | True              | True                   | lr1e-3_tau1.0_waqasT_focalT          |
# | 14  | 1e-3                  | 1.0             | True              | False                  | lr1e-3_tau1.0_waqasT_focalF          |
# | 15  | 1e-3                  | 1.0             | False             | True                   | lr1e-3_tau1.0_waqasF_focalT          |
# | 16  | 1e-3                  | 1.0             | False             | False                  | lr1e-3_tau1.0_waqasF_focalF          |
# | 17  | 5e-3                  | 0.33            | True              | True                   | lr5e-3_tau0.33_waqasT_focalT         |
# | 18  | 5e-3                  | 0.33            | True              | False                  | lr5e-3_tau0.33_waqasT_focalF         |
# | 19  | 5e-3                  | 0.33            | False             | True                   | lr5e-3_tau0.33_waqasF_focalT         |
# | 20  | 5e-3                  | 0.33            | False             | False                  | lr5e-3_tau0.33_waqasF_focalF         |
# | 21  | 5e-3                  | 1.0             | True              | True                   | lr5e-3_tau1.0_waqasT_focalT          |
# | 22  | 5e-3                  | 1.0             | True              | False                  | lr5e-3_tau1.0_waqasT_focalF          |
# | 23  | 5e-3                  | 1.0             | False             | True                   | lr5e-3_tau1.0_waqasF_focalT          |
# | 24  | 5e-3                  | 1.0             | False             | False                  | lr5e-3_tau1.0_waqasF_focalF          |




# | Run | DEFAULT_LEARNING_RATE | DEFAULT_LSE_TAU | DEFAULT_WAQAS_WAY | DEFAULT_USE_FOCAL_LOSS | k8s (job name hint)                  |
# | --- | --------------------- | --------------- | ----------------- | ---------------------- | ------------------------------------ |
# | 1   | 2e-4                  | 0.33            | True              | True                   | lr2e-4_tau0.33_waqasT_focalT         |
# | 2   | 2e-4                  | 0.33            | True              | False                  | lr2e-4_tau0.33_waqasT_focalF         |

# | 5   | 2e-4                  | 1.0             | True              | True                   | lr2e-4_tau1.0_waqasT_focalT          |
# | 6   | 2e-4                  | 1.0             | True              | False                  | lr2e-4_tau1.0_waqasT_focalF          |

# | 9   | 1e-3                  | 0.33            | True              | True                   | lr1e-3_tau0.33_waqasT_focalT         |
# | 10  | 1e-3                  | 0.33            | True              | False                  | lr1e-3_tau0.33_waqasT_focalF         |
# | 11  | 1e-3                  | 0.33            | False             | True                   | lr1e-3_tau0.33_waqasF_focalT         |
# | 12  | 1e-3                  | 0.33            | False             | False                  | lr1e-3_tau0.33_waqasF_focalF         |

# | 13  | 1e-3                  | 1.0             | True              | True                   | lr1e-3_tau1.0_waqasT_focalT          |
# | 14  | 1e-3                  | 1.0             | True              | False                  | lr1e-3_tau1.0_waqasT_focalF          |

# | 17  | 5e-3                  | 0.33            | True              | True                   | lr5e-3_tau0.33_waqasT_focalT         |
# | 18  | 5e-3                  | 0.33            | True              | False                  | lr5e-3_tau0.33_waqasT_focalF         |

# | 21  | 5e-3                  | 1.0             | True              | True                   | lr5e-3_tau1.0_waqasT_focalT          |
# | 22  | 5e-3                  | 1.0             | True              | False                  | lr5e-3_tau1.0_waqasT_focalF          |


print("DEBUG: Single-GPU launcher script is starting...")

# =============================================================================
# EASY CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# =============================================================================
# Simply change these values below to modify the default behavior:

DEFAULT_LEARNING_RATE = 2e-4        # Learning rate for training
DEFAULT_LSE_TAU = 0.33                # Temperature parameter for LSE aggregation
DEFAULT_WAQAS_WAY = True             # Set to True to use Waqas attention aggregation
FOCAL_LOSS = True                  # Set to True to use focal loss

DEFAULT_BALANCED_SAMPLING = False     # Set to True to use balanced sampling
CLASS_WEIGHTS = True                  # Set to True to use class weights


def get_output_dir_name():
    return "_".join([
        "waqas" if DEFAULT_WAQAS_WAY else "lse",
        f"tau{DEFAULT_LSE_TAU}",
        "balanced" if DEFAULT_BALANCED_SAMPLING else "unbalanced",
        "weighted" if CLASS_WEIGHTS else "unweighted",
        f"lr{DEFAULT_LEARNING_RATE}",
        "focal" if FOCAL_LOSS else "ce"
    ])
OUTPUT_DIR = f"output1_ALL_ME_{get_output_dir_name()}"  

# You can still override these defaults via command line:
# python run_fineTune_single_gpu_launcher.py --waqas-way --lse-tau 0.5 --balanced-sampling True --lr 2e-4
# =============================================================================

print(f"CONFIGURATION:")
print(f"  Waqas Way: {DEFAULT_WAQAS_WAY}")
print(f"  LSE Tau: {DEFAULT_LSE_TAU}")
print(f"  Balanced Sampling: {DEFAULT_BALANCED_SAMPLING}")
print(f"  Learning Rate: {DEFAULT_LEARNING_RATE}")
print(f"  Class Weights: {CLASS_WEIGHTS}")
print(f"  Output Directory: {OUTPUT_DIR}")
print("=" * 50)

# Log the configuration as well
logging.info("=" * 50)
logging.info("LAUNCHER CONFIGURATION")
logging.info("=" * 50)
logging.info(f"Waqas Way: {DEFAULT_WAQAS_WAY}")
logging.info(f"LSE Tau: {DEFAULT_LSE_TAU}")
logging.info(f"Balanced Sampling: {DEFAULT_BALANCED_SAMPLING}")
logging.info(f"Learning Rate: {DEFAULT_LEARNING_RATE}")
logging.info(f"Class Weights: {CLASS_WEIGHTS}")
logging.info(f"Focal Loss: {FOCAL_LOSS}")
logging.info(f"Output Directory: {OUTPUT_DIR}")
logging.info("=" * 50)

LOG_DIR = "/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "gpu_info.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def log_gpu_info():
    """Log GPU information for single-GPU setup"""
    # How many CUDA-visible devices?
    ngpu = torch.cuda.device_count()
    logging.info(f"CUDA_VISIBLE_DEVICES reports {ngpu} device(s).")

    for i in range(ngpu):
        props = torch.cuda.get_device_properties(i)
        # Major/minor, total memory, multiProcessor count, name
        logging.info(
            f"  – GPU {i}: {props.name}  "
            f"(SM {props.major}.{props.minor}, "
            f"{props.total_memory/1024**3:.1f} GB VRAM, "
            f"{props.multi_processor_count} SMs)"
        )

    # Log CUDA and cuDNN versions from PyTorch
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA build version: {torch.version.cuda}")
    logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    # If you want the actual driver, MIG, topology info via nvidia-smi:
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,mig.mode.current", "--format=csv,noheader"]
        ).decode("utf-8")
        logging.info("`nvidia-smi --query-gpu=` output:\n" + smi.strip())
    except Exception as e:
        logging.warning(f"Could not run nvidia-smi: {e}")


def install_packages():
    """Install required packages"""
    commands = [
        ["pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu117", "torch==2.0.0", "torchvision==0.15.0", "omegaconf", "torchmetrics==0.10.3", "fvcore", "iopath", "xformers==0.0.18", "submitit","numpy<2.0"],
        ["pip", "install", "--extra-index-url", "https://pypi.nvidia.com", "cuml-cu11"],
        ["pip", "install", "black==22.6.0", "flake8==5.0.4", "pylint==2.15.0"],
        ["pip", "install", "mmsegmentation==0.27.0"],
        ["pip", "install", "mmcv-full==1.5.0"]
    ]
    for i, command in enumerate(commands):
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


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

def prepare_balanced_validation(csv_path, output_csv_path):
    """Prepare a balanced validation set from the original data"""
    df = pd.read_csv(csv_path)
    
    # Print original validation set size
    print("Original validation set size:", df[df["split"]=="val"].shape)

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
    df_new.to_csv(output_csv_path, index=False)
    return output_csv_path


def main(args):
    """Main function to run single-GPU training"""
    # Install required packages if needed
    if args.install_packages:
        install_packages()
    
    # Prepare balanced validation set if needed
    if args.balance_val:
        csv_path = prepare_balanced_validation(
            args.csv, 
            args.csv.replace('.csv', '_balanced.csv')
        )
    else:
        csv_path = args.csv

    os.makedirs(args.metrics_dir, exist_ok=True)

    # Auto-generate label-cols from groups and binary-names
    try:
        parsed_groups = parse_groups_list(args.groups)
        # Flatten groups and combine with binary names, removing duplicates
        label_cols = list(dict.fromkeys([label for group in parsed_groups for label in group] + args.binary_names))
        print(f"Auto-generated label-cols: {label_cols}")
    except ValueError as e:
        print(f"Error parsing groups: {e}")
        return

    # Build the command for single-GPU training (no torchrun needed)
    python_command = [
        "python3",
        args.script_path,
        "--csv", csv_path,
        "--split-col", args.split_col,
        "--accum-steps", str(args.accum_steps),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--optimizer", args.optimizer,
        "--num-attn-heads", str(args.num_attn_heads),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),

        "--output", args.output,
        "--print-every", str(args.print_every),
        "--val-every", str(args.val_every),
        "--metrics-dir", args.metrics_dir,
        "--warmup-steps", str(args.warmup_steps),
        "--max-chunks", str(args.max_chunks),
        "--lse-tau", str(args.lse_tau),
        "--groups", args.groups,
        "--group-names",
    ] + args.group_names
    
    # Only add binary-names if there are any
    if args.binary_names:
        python_command.extend(["--binary-names"] + args.binary_names)
    
    # Add boolean flags if enabled
    if args.class_weights == "True":
        python_command.append("--class-weights")
        python_command.append("True")
    
    # Add balanced sampling flag if enabled
    if args.balanced_sampling == "True":
        python_command.append("--balanced-sampling")
        python_command.append("True")
    
    # Add no-load-pretrained flag if enabled
    if args.no_load_pretrained:
        python_command.append("--no-load-pretrained")
        print(f"DEBUG: Added --no-load-pretrained flag to command")
    else:
        print(f"DEBUG: No --no-load-pretrained flag (args.no_load_pretrained = {args.no_load_pretrained})")
    
    # Add no-freeze-backbone flag if user wants to train all parameters
    if args.no_freeze_backbone:
        python_command.append("--no-freeze-backbone")
        print(f"DEBUG: Added --no-freeze-backbone flag to command (backbone will be trainable)")
    else:
        print(f"DEBUG: No --no-freeze-backbone flag (backbone will be frozen by default)")
    
    # Add no-learnable-weights flag if user wants to disable learnable weights
    if args.no_learnable_weights:
        python_command.append("--no-learnable-weights")
        print(f"DEBUG: Added --no-learnable-weights flag to command (using equal weights)")
    else:
        print(f"DEBUG: No --no-learnable-weights flag (learnable weights enabled by default)")
    
    # Add waqas-way flag if enabled
    if args.waqas_way:
        python_command.append("--waqas-way")
        print(f"DEBUG: Added --waqas-way flag to command")
    else:
        print(f"DEBUG: No --waqas-way flag (using default attention aggregation)")
    
    # Add focal loss flag if enabled
    if args.use_focal_loss:
        python_command.append("--use-focal-loss")
        print(f"DEBUG: Added --use-focal-loss flag to command")
    else:
        print(f"DEBUG: No --use-focal-loss flag (using standard cross entropy)")
    
    # Add loss-wrapper argument
    python_command.extend(["--loss-wrapper", args.loss_wrapper])
    print(f"DEBUG: Added --loss-wrapper {args.loss_wrapper} to command")
    
    print(f"Running single-GPU command: {' '.join(python_command)}")
    
    # Execute the training (no distributed environment needed)
    subprocess.run(python_command, check=True)

    print(f"Training metrics will be saved to: {args.metrics_dir}/training_metrics_single_gpu_*.jsonl")


def run_training():
    """Parse arguments and run single-GPU training"""
    parser = argparse.ArgumentParser(description="Launch Single-GPU Training")
    
    # Data parameters
    parser.add_argument("--csv", type=str, 
                       default='/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/CSVs/ready_finetune_MNT_info_with_flags.csv', 
                       help="Path to the CSV file containing dataset information")
    
    # Group definitions for mutually exclusive tasks
    parser.add_argument("--groups", type=str,
                       default="[['is_M0', 'is_M1A', 'is_M1b_M1c'], ['is_N0_N1', 'is_N2', 'is_N3'], ['is_T0_T1', 'is_T2', 'is_T3_T4']]",
                       help="List of groups in Python list format. Each inner list contains mutually exclusive labels that sum to 1. Format: '[[a,b,c], [d], [e,f,g]]'. Label-cols will be auto-generated from groups + binary-names.")
    
    # Override group names to match your prediction tasks
    parser.add_argument("--group-names", type=str, nargs='+', 
                       default=['metastasis', 'node', 'tumor'],
                       help="Names for the groups (must match number of groups)")
    
    # Binary (independent) task names
    parser.add_argument("--binary-names", type=str, nargs='+', 
                       default=['is_mw'],
                       help="Independent binary task names. Label-cols will be auto-generated from groups + binary-names.")
    parser.add_argument("--balance-val", action="store_true", 
                        help="Whether to balance the validation set")
    parser.add_argument("--class-weights", type=str, default=str(CLASS_WEIGHTS), choices=["True", "False"],
                        help="Whether to use class weights in the loss function (True/False, default: True)")
    parser.add_argument("--balanced-sampling", type=str, default=str(DEFAULT_BALANCED_SAMPLING), choices=["True", "False"],
                        help=f"Whether to use balanced sampling for training (True/False, default: {DEFAULT_BALANCED_SAMPLING})")
    parser.add_argument("--use-focal-loss", action="store_true", default=FOCAL_LOSS,
                        help="Whether to use focal loss instead of cross entropy for group tasks")
    
    # Hardware parameters
    parser.add_argument("--num-workers", type=int, default=2, 
                        help="Number of workers for data loading")
    
    # Training parameters
    parser.add_argument("--accum-steps", type=int, default=64, 
                       help="Number of steps to accumulate gradients over")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Actual batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=200, 
                       help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, 
                       help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--weight-decay", type=float, default=1e-2, 
                       help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", 
                       choices=["adam", "adamw", "sgd"], 
                       help="Optimizer to use")
    
    
    # Model parameters
    parser.add_argument("--num-attn-heads", type=int, default=3, 
                       help="Number of attention heads in the transformer aggregator")
    parser.add_argument("--num-layers", type=int, default=1, 
                       help="Number of layers in the transformer aggregator")
    parser.add_argument("--dropout", type=float, default=0.6, 
                       help="Dropout rate in the transformer aggregator")
    parser.add_argument("--max-chunks", type=int, default=64, 
                       help="Maximum number of chunks to process per sample")
    
    # Output parameters
    parser.add_argument("--output", type=str, 
                       default=os.path.join("/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/output/outputSep25_MNT/", OUTPUT_DIR),
                       help="Output directory for logs and checkpoints")
    parser.add_argument("--print-every", type=int, default=500, 
                       help="Print training stats every N steps")
    parser.add_argument("--val-every", type=int, default=2000, 
                       help="Run validation every N steps")
    
    # Setup parameters
    parser.add_argument("--install-packages", action="store_true",
                       help="Whether to install required packages")
    
    # Add metrics directory parameter
    parser.add_argument("--metrics-dir", type=str, 
                       default="/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/metrics",
                       help="Directory to save training metrics")
    
    # New parameter for warmup steps
    parser.add_argument("--warmup-steps", type=int, default=5, 
                       help="Number of steps for warmup")
    
    # New parameter for no-load-pretrained
    parser.add_argument("--no-load-pretrained", action="store_true",
                        help="Whether to load pretrained model (default: False)")
    parser.add_argument("--no-freeze-backbone", action="store_true",
                        help="Disable freezing the DINO backbone (train all parameters) - DEFAULT: backbone is frozen")
    parser.add_argument("--no-learnable-weights", action="store_true",
                        help="Disable learnable task weights (use equal weights) - DEFAULT: learnable weights are enabled")
    parser.add_argument("--loss-wrapper", type=str, choices=["multitask"], default="multitask",
                        help="Use UncertaintyWeightedSumNamed for learnable task weights (default: enabled)")
    parser.add_argument("--split-col", type=str, default="split", 
                       help="Column in CSV to use for train/val splits")
    parser.add_argument("--lse-tau", type=float, default=DEFAULT_LSE_TAU,
                       help=f"Temperature parameter for LSE (Log-Sum-Exp) aggregation (default: {DEFAULT_LSE_TAU})")
    parser.add_argument("--waqas-way", action="store_true", default=DEFAULT_WAQAS_WAY,
                       help=f"Use Waqas way for attention aggregation (default: {DEFAULT_WAQAS_WAY})")

    # Path to the single-GPU training script
    parser.add_argument("--script-path", type=str, 
                        default="/rsrch7/home/ip_rsrch/wulab/Lung_Foundation_Model_Data_/Down-stream_tasks/Histology/FineTune/code_1gpu_loss_wrapper_muex/2_run_fineTune_single_gpu_learnable_loss-wrapper.py", #2_run_fineTune_single_gpu_learnable.py",
                        help="Path to the single-GPU training script")

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"DEBUG: Single-GPU launcher received args.no_load_pretrained = {args.no_load_pretrained}")
    print(f"DEBUG: All launcher args: {vars(args)}")
    
    log_gpu_info()
    main(args) 


if __name__ == "__main__":
    run_training()






#     parser.add_argument("--balanced-sampling", type=str, default="False", choices=["True", "False"],
#                         help="Whether to use balanced sampling for training (True/False, default: True)")

# parser.add_argument("--lr", type=float, default=1e-4, 
#                    help="Learning rate")
