#!/bin/sh
#SBATCH --output=output_%%%NAME%%%_%j.out
#SBATCH --error=error_%%%NAME%%%_%j.er
#SBATCH --job-name=%%%NAME%%%

repositorypath="/PATH/TO/REPOSITORY"

export PYTHONPATH=$PYTHONPATH:${repositorypath}

# Set NEST enviornment
source "/PATH/TO/NEST/INSTALLATION/bin/nest_vars.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python ${repositorypath}/experiments/run_tasks.py --num_threads ${OMP_NUM_THREADS} %%%PARAMS%%%