"""
Postproecssing of ABL simulations in Nek5000
Linnea Huusko, 2024-03-08
"""

from abltools import (
    plot_diagnostics,
    plot_wmles_diagnostics,
    plot_history_points,
    plot_vertical_profiles,
    get_averages,
)
import xarray as xr
import sys

path = "."
casename = sys.argv[1]
label = sys.argv[2]
ref_case = sys.argv[3]

# --- Diagnostics ----------------------------------------------------------------------
fig_diag, diag = plot_diagnostics(path)
fig_diag.savefig(path + "/figures/diagnostics.png", bbox_inches="tight")
diag_averages = get_averages(diag)

# --- WMLES diagnostics ----------------------------------------------------------------
fig_wmles, wmles = plot_wmles_diagnostics(path)
fig_wmles.savefig(path + "/figures/wmles.png", bbox_inches="tight")
wmles_averages = get_averages(wmles)

# --- History points -------------------------------------------------------------------
fig_his, his = plot_history_points(path + f"/{casename}.his")
fig_his.savefig(path + "/figures/history_points.png", bbox_inches="tight")

# --- Profiles -------------------------------------------------------------------------
ds = xr.open_dataset(path + f"/data/{casename}.nc")
fig_profiles = plot_vertical_profiles(ds, label, ref_case)

with open("diagnostics_summary.dat", "w") as file:
    for key, value in diag_averages.items():
        if key != "time":
            file.write(f"{key:<10}{value:2.3f}\n")
    for key, value in wmles_averages.items():
        if key != "time":
            file.write(f"{key:<10}{value:2.3f}\n")
