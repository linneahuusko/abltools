#!/bin/bash -l
#SBATCH -J postprocessing
#SBATCH -t 0:30:00
#SBATCH --nodes 1
#SBATCH -p main
#SBATCH -A naiss2023-1-5
#SBATCH --mail-type=BEGIN,END,FAIL

conda activate Nek5000-Python-env

casename="n_32"
N=32
python scripts/time_average.py $casename $N > time_average.log

python scripts/plot_diagnostics.py $casename "n32 TKE" "neutral" > plotting_diagnostics.log
