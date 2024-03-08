# Postprocessing scripts

Scripts to be used in a case directory.

`postprocessing.sh` can be called following the completion of a run, and will call Python scripts to compute averages and plot timeseries and averages.

`plot_diagnostics.py` plots timeseries and averages, and outputs a text file with some diagnostics.

`time_average.py` computes time averages from statistics files, using methods from `postcipes`.
