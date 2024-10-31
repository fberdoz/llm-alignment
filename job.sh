#!/bin/bash

# SLURM directives
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/fberdoz/net_scratch/llm-alignment/jobs/%j.out
#SBATCH --error=/itet-stor/fberdoz/net_scratch/llm-alignment/jobs/%j.err
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#Comment#SBATCH --exclude=tikgpu10,tikgpu[06-09]
#SBATCH --nodelist=tikgpu10
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'

# Exit on errors
set -o errexit

# Record the start time
START_TIME=$(date +%s)

# Define variables
ETH_USERNAME=${USER}
CONDA_ENVIRONMENT=pytcu11
PROJECT_NAME=llm-alignment
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}

# Create the jobs directory if it doesn't exist
mkdir -p "${DIRECTORY}/jobs"
cd "${DIRECTORY}" || exit 1


# Set up wandb directories on local scratch
WANDB_DIR="/scratch/${USER}/wandb_dir_{$SLURM_JOB_ID}"
WANDB_CACHE_DIR="${WANDB_DIR}/.cache"
export WANDB_DIR WANDB_CACHE_DIR
mkdir -vp "${WANDB_CACHE_DIR}" || { echo "Error: Failed to create cache directory."; exit 1; }


# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# Activate conda environment
if [[ -f /itet-stor/${USER}/net_scratch/conda/bin/conda ]]; then
    eval "$(/itet-stor/${USER}/net_scratch/conda/bin/conda shell.bash hook)"
    conda activate "${CONDA_ENVIRONMENT}"
    echo "Conda environment '${CONDA_ENVIRONMENT}' activated"
else
    echo "Conda not found at /itet-stor/${USER}/net_scratch/conda/bin/conda" >&2
    exit 1
fi

# Execute your code with the config file
echo "------- Running main.py --------"
python main.py --config config.yaml
echo "--- Finished running main.py ---"

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# Record  and display execution time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS_REMAINING=$((ELAPSED_TIME % 60))
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS_REMAINING}s"

# Move and cleanup wandb data
# Remove cache directory
rm -r "${WANDB_CACHE_DIR}" || { echo "Error: Failed to remove cache directory."; exit 1; }

# Ensure the wandb directory inside the project folder exists
mkdir -p "${DIRECTORY}/wandb" || { echo "Error: Failed to create project wandb directory."; exit 1; }

# Find the run_X directory
RUN_FOLDER=$(find "${WANDB_DIR}/wandb/" -maxdepth 1 -type d -name "run*" -print -quit)

# Check if we found the folder
if [ -z "$RUN_FOLDER" ]; then
    echo "Error: No directory starting with 'run' found in ${WANDB_DIR}/wandb"
    exit 1
fi

# Move the found run_X folder to the project directory
mv -v "$RUN_FOLDER" "${DIRECTORY}/wandb/" || { echo "Error: Failed to move $RUN_FOLDER."; exit 1; }

# Success message
echo "Successfully moved $RUN_FOLDER to ${DIRECTORY}/wandb/"

# Remove the wandb directory from local scratch
rm -r "${WANDB_DIR}"

# End the script with exit code 0
exit 0
