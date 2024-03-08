#!/bin/bash -l
#SBATCH -J postprocessing
#SBATCH -t 0:30:00
#SBATCH --nodes 1
#SBATCH -p main
#SBATCH -A naiss2023-1-5
#SBATCH --mail-type=BEGIN,END,FAIL

conda activate Nek5000-Python-env
python scripts/time_average.py > time_average.log

python plotting_diagnostics.py "n_32 TKE" neutral > plotting_diagnostics.log
