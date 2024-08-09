#!/bin/bash

# Job name
#SBATCH --job-name=salt

# choose the GPU queue
#SBATCH --qos=regular
#SBATCH -C gpu
#SBATCH -A atlas

# request nodes
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -t 24:00:00


# note! this needs to match --trainer.devices!
#SBATCH --ntasks-per-node=4

# request enough memory
##SBATCH --mem=100G

# Change log names; %j gives job id, %x gives job name
#SBATCH --output=/pscratch/sd/n/nishank/shapiro_pi2/job_output/slurm-%j.%x.out
# optional separate error output file
# activate environment
source /pscratch/sd/n/nishank/shapiro_pi2/salt/setup/setup_conda.sh
conda activate salt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Activated environment ${CONDA_DEFAULT_ENV}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# run the training
echo "Running training script..."
srun salt fit --config /pscratch/sd/n/nishank/shapiro_pi2/salt/salt/configs/GN2_charge_reco.yaml
