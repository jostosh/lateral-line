#!/usr/bin/env bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=LLSWEEP
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output LLSWEEP-%j.log
#SBATCH --mem=2000
#SBATCH --partition=short

module load tensorflow
source $HOME/envs/ll/bin/activate
cd $HOME/lateral-line

srun python3 ./train.py