#!/bin/bash
#SBATCH --job-name=Mixup_vs_CutMix_Exp
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=outputs/%x/%x_%j_%a.log
#SBATCH --partition=kisski
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:A100:1
#SBATCH --array=1-3  # 3 experiments: Baseline, MixUp only, CutMix only

set -euo pipefail  # Strict mode

echo "=== Job starting at $(date) on $(hostname) ==="
echo "Working directory: $(pwd)"

# --- EXPERIMENT CONFIGURATION ---
# Define the configurations to test. The array index will select one.
# Format is: "mixup_alpha cutmix_alpha run_name"
configs=(
    "0.0 0.0 baseline"
    "0.2 0.0 mixup_only"
    "0.0 0.2 cutmix_only"
)

# Get the configuration for the current Slurm array task ID
current_config=(${configs[$((SLURM_ARRAY_TASK_ID - 1))]})
CURRENT_MIXUP=${current_config[0]}
CURRENT_CUTMIX=${current_config[1]}
RUN_NAME=${current_config[2]}

# --- LOGGING AND DIRECTORY SETUP ---
# Use the winning architecture settings as the base for the output directory
OUTPUT_DIR="./output/regularization_exp/${RUN_NAME}"

echo "------------------------------------------------"
echo "SLURM JOB: ${SLURM_JOB_ID}, ARRAY TASK: ${SLURM_ARRAY_TASK_ID}"
echo "RUN NAME: ${RUN_NAME}"
echo "PARAMETERS:"
echo "  - MixUp Alpha: ${CURRENT_MIXUP}"
echo "  - CutMix Alpha: ${CURRENT_CUTMIX}"
echo "------------------------------------------------"


# --- ENVIRONMENT SETUP ---
echo "Loading modules..."
ml load gcc/13.2.0-nvptx || { echo "[ERROR] Failed to load gcc/13.2.0-nvptx" >&2; exit 1; }
ml load gcc/13.2.0 || { echo "[ERROR] Failed to load gcc/13.2.0" >&2; exit 1; }
ml load cuda || { echo "[ERROR] Failed to load cuda" >&2; exit 1; }

VENV_PATH="/mnt/vast-kisski/projects/kisski-mhh-snnergy/multi-modal/.venv/bin/activate"
if [[ -f "$VENV_PATH" ]]; then
    source "$VENV_PATH"
else
    echo "[ERROR] Virtual environment not found at $VENV_PATH" >&2
    exit 1
fi

WORKDIR="/mnt/vast-kisski/projects/kisski-mhh-snnergy/multi-modal"
if [[ -d "$WORKDIR" ]]; then
    cd "$WORKDIR"
else
    echo "[ERROR] Working directory not found at $WORKDIR" >&2
    exit 1
fi

# --- RUN TRAINING ---
echo "=== Starting training with best architecture ==="
python train_video.py \
    --data-path ./data/crema/Crema_frames_8_90_cropped_128 \
    --output-dir "${OUTPUT_DIR}" \
    --img-size 128 \
    --mlp-ratios 2 \
    --depths 2 4 8 \
    --num-heads 2 4 6 \
    --model qkf_snn \
    --epochs 120 \
    --batch-size 32 \
    --opt adamw \
    --sched cosine \
    --warmup-epochs 20 \
    --patience-epochs 40 \
    --time-step 8 \
    --lr 0.0005 \
    --weight-decay 0.05 \
    --mixup ${CURRENT_MIXUP} \
    --cutmix ${CURRENT_CUTMIX}

echo "=== Training finished at $(date) ==="
echo "--- Task Finished ---"
