#!/bin/bash
#SBATCH --job-name=train_regid
#SBATCH --partition=dev
#SBATCH --account=MST113097
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/train_regid_%j.out
#SBATCH --error=logs/train_regid_%j.err

set -e

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "========================================"

PROJECT_DIR=/work/$(whoami)/sam-3d-objects/lab_test_for_python/try_ablation
SIF_PATH=/work/$(whoami)/sam_3d_env/base.sif
CONDA_PATH=/work/$(whoami)/my_conda/bin/activate
CONFIG_PATH=configs/train_regid.yaml

cd ${PROJECT_DIR}
mkdir -p logs

echo "Current directory outside singularity:"
pwd

echo "GPU status outside singularity:"
nvidia-smi || true

echo "========================================"
echo "Entering Singularity"
echo "SIF_PATH: ${SIF_PATH}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "========================================"

singularity exec --nv -B /work ${SIF_PATH} bash -lc "
    source ${CONDA_PATH}
    conda activate sam3d-objects

    unset SSL_CERT_FILE
    unset SSL_CERT_DIR
    export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH

    cd ${PROJECT_DIR}

    echo '========================================'
    echo 'Inside Singularity'
    echo 'Current directory:'
    pwd

    echo 'Python path:'
    which python

    echo 'Python version:'
    python --version

    echo 'Conda prefix:'
    echo \$CONDA_PREFIX

    echo 'GPU status inside singularity:'
    nvidia-smi || true

    echo 'Start rigid/global alignment training...'
    echo 'Config: ${CONFIG_PATH}'
    echo '========================================'

    python train_regid.py --config ${CONFIG_PATH}
"

echo "========================================"
echo "Rigid/global alignment training finished"
echo "End time: $(date)"
echo "========================================"