"""
Postproecssing of ABL simulations in Nek5000
Linnea Huusko, 2024-03-08
"""

from Plotting_functions import (
    plot_diagnostics,
    plot_wmles_diagnostics,
    plot_history_points,
    plot_vertical_profiles,
)
import xarray as xr
import sys

path = "."
label = sys.argv[0]
ref_case = sys.argv[1]

# --- Diagnostics ----------------------------------------------------------------------
fig_diag, diag = plot_diagnostics(path)
fig_diag.savefig(path + "/figures/diagnostics.png", bbox_inches="tight")

# --- WMLES diagnostics ----------------------------------------------------------------
fig_wmles, wmles = plot_wmles_diagnostics(path)
fig_wmles.savefig(path + "/figures/wmles.png", bbox_inches="tight")

# --- History points -------------------------------------------------------------------
fig_his, his = plot_history_points(path)
fig_his.savefig(path + "/figures/history_points.png", bbox_inches="tight")

# --- Profiles -------------------------------------------------------------------------
ds = xr.open_dataset(path + f"/data/stats.nc")
fig_profiles = plot_vertical_profiles(ds, label, ref_case)
