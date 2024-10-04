"""
Functions for plotting output from Nek5000 ABL simulations
Linnea Huusko, 2024-03-08
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum_hdf5(
    f,
    path: str,
    variable: str,
    dimension: str,
    height: float,
    t0: int,
    axis=None,
    label=None,
    inertial: bool = False,
    **kwargs,
) -> tuple[float, float]:
    """Plot a spectrum of a given variable in the given dimension and return spectrum and frequencies

    Parameters:
        f (h5py.File):      h5py file, output from nektsrs
        path (str):         path to experiment directory
        variable (str):     variable for which to calculate the spectrum ("u", "v", "w", or "theta")
        dimension (str):    dimension over which to calculate the spectrum ("x", "z", or "t")
        height (float):     height (z, in m) at which to compute the spectrum
        t0 (int):           initial time, first time step (index) to be included
        axis (ax):          axis to plot spectrum on (if not None)
        label (str):        label for figure legend (if not None and axis not None)
        intertial (bool):   if True, plot -5/3 line in inertial subrange (default is False)

    Returns:
        s ():
        p ():

    """
    from scipy import signal
    import h5py

    locations = np.loadtxt(f"{path}/Point_locations.txt", delimiter=",", max_rows=1)
    shape = np.insert(locations, 0, len(f["t"][:])).astype(int).tolist()

    if variable == "u":
        var_index = 0
    elif variable == "v":
        var_index = 1
    elif variable == "w":
        var_index = 2
    elif variable == "theta":
        var_index = 4

    u = f["data"][:, var_index, :].reshape(
        shape
    )  # u now used as dummy for u, v, w, theta

    locs_x = f["locs"][:, 0].reshape(shape[1:])[:, 0, 0]
    locs_y = f["locs"][:, 1].reshape(shape[1:])[0, :, 0]
    locs_z = f["locs"][:, 2].reshape(shape[1:])[0, 0, :]

    height_diff = np.absolute(locs_y - height)

    height_index = np.argmin(height_diff)

    U = u[t0:, :, height_index, :]

    if dimension == "x":
        delta = locs_x[1] - locs_x[0]
        s, p = signal.periodogram(
            U,
            (1 / delta),
            scaling="density",
            axis=1,
            detrend="constant",
            window="boxcar",
        )
        p = p.mean(axis=-1).mean(axis=0)
    elif dimension == "z":
        delta = locs_z[1] - locs_z[0]
        s, p = signal.periodogram(
            U,
            (1 / delta),
            scaling="density",
            axis=-1,
            detrend="constant",
            window="boxcar",
        )
        p = p.mean(axis=1).mean(axis=0)
    elif dimension == "t":
        delta = f["t"][1] - f["t"][0]
        s, p = signal.periodogram(
            U,
            (1 / delta),
            scaling="density",
            axis=0,
            detrend="constant",
            window="boxcar",
        )
        p = p.mean(axis=1).mean(axis=-1)

    if axis:
        axis.plot(s, p * s, label=label, **kwargs)

    if axis and inertial:
        axis.plot(
            s[10:],
            s[10:] ** (-5 / 3) / 30000,
            linewidth=1,
            linestyle="dashed",
            color="black",
        )

    return s, p


def plot_spectrum_netcdf(
    ds,
    variable: str,
    dimension: str,
    height: float,
    t0: int,
    t1: int,
    axis=None,
    label=None,
    scale=False,
    inertial: bool = False,
    **kwargs,
) -> tuple[float, float]:
    """Plot a spectrum of a given variable in the given dimension and return spectrum and frequencies

    Parameters:
        ds (xarray dataset):        netCDF file with processed output from nektsrs
        path (str):                 path to experiment directory
        variable (str):             variable for which to calculate the spectrum ("u", "v", "w", or "theta")
        dimension (str):            dimension over which to calculate the spectrum ("x", "z", or "t")
        height (float):             height (z, in m) at which to compute the spectrum
        t0 (int):                   time at which to start spectrum calculation
        t1 (int):                   time at which to end spectrum calculation
        axis (ax):                  axis to plot spectrum on (if not None)
        label (str):                label for figure legend (if not None and axis not None)
        intertial (bool):           if True, plot -5/3 line in inertial subrange (default is False)

    Returns:
        s ():
        p ():

    """
    from scipy import signal

    delta = (ds[dimension][1] - ds[dimension][0]).values

    if dimension == "x":
        s, p = signal.periodogram(
            ds[variable].sel(y=height, method="nearest").sel(time=slice(t0, t1)),
            (1 / delta),
            scaling="density",
            axis=1,
            detrend="constant",
            window="boxcar",
        )
        p = p.mean(axis=-1).mean(axis=0)

    elif dimension == "z":
        s, p = signal.periodogram(
            ds[variable].sel(y=height, method="nearest").sel(time=slice(t0, t1)),
            (1 / delta),
            scaling="density",
            axis=-1,
            detrend="constant",
            window="boxcar",
        )
        p = p.mean(axis=1).mean(axis=0)
    elif dimension == "t":
        s, p = signal.periodogram(
            ds[variable]
            .sel(y=height, method="nearest")(1 / delta)
            .sel(time=slice(t0, t1)),
            scaling="density",
            axis=0,
            detrend="constant",
            window="boxcar",
        )
        p = p.mean(axis=1).mean(axis=-1)

    if axis and scale:
        # print("Scaling!")
        axis.loglog(s, p * s, label=label, **kwargs)

        if inertial:
            axis.plot(
                s[10:],
                s[10:] ** (-2 / 3) / 30000,
                linewidth=1,
                linestyle="dashed",
                color="black",
            )

    if axis and not scale:
        # print("Not scaling!")
        axis.loglog(s, p, label=label, **kwargs)

        if inertial:
            axis.plot(
                s[10:],
                s[10:] ** (-5 / 3) / 500,
                linewidth=1,
                linestyle="dashed",
                color="black",
            )

    return s, p


#### Vertical Profiles #################################################################
def plot_vertical_profiles(
    datasets: list,
    labels: list,
    case: str,
    plot_ref=True,
    color="C0",
    normalize_BLH=False,
):
    """
    Create figure and plot vertical profiles of a number of variables
    for a given ABL case.

    Parameters:
        datasets:                   list of xarray datasets with output from time_average.py
        label (str):                list of labels for legend
        case (str):                 name of case for reading Peter Sullivans reference data
        plot_ref (bool):            if True, plot reference curves from NCAR model
        color (str):                color for plotting the profiles
        normalize_BLH (bool):       if True, normalize profiles against BLH, if False plot
        against absolute height

    Returns:
        fig:                    figure with all plots
    """

    from abltools import read_profile_from_ref, read_BLH_from_ref

    if case == "stable" or case == "neutral":
        t_ref = 265
    elif case == "mixed" or case == "free_conv":
        t_ref = 300

    fig, [
        [ax_u, ax_uw, ax_vw],
        [ax_uu, ax_vv, ax_ww],
        [ax_nutot, ax_w_skew, ax_dTdz],
        [ax_T, ax_TT, ax_Tv],
    ] = plt.subplots(4, 3, figsize=(12, 16), sharey=True)

    # --- Reference --------------------------------------------------------------------
    if plot_ref:
        if normalize_BLH:
            z_i_ref = read_BLH_from_ref(case)
        else:
            z_i_ref = 1
        u_ref, z_ref, _ = read_profile_from_ref("UXYM", case)
        ax_u.plot(u_ref, z_ref / z_i_ref, color="black", label="ref")

        v_ref, z_ref, _ = read_profile_from_ref("VXYM", case)
        ax_u.plot(v_ref, z_ref / z_i_ref, color="black", linestyle="dashed")

        uu_ref, z_ref, _ = read_profile_from_ref("UPS", case)
        ax_uu.plot(uu_ref, z_ref / z_i_ref, color="black", label="ref")

        vv_ref, z_ref, _ = read_profile_from_ref("VPS", case)
        ax_vv.plot(vv_ref, z_ref / z_i_ref, color="black", label="ref")

        ww_ref, z_ref, _ = read_profile_from_ref("WPS", case)
        ax_ww.plot(ww_ref, z_ref / z_i_ref, color="black", label="ref")

        uw_ref_res, z_ref, _ = read_profile_from_ref("UWLE", case)
        ax_uw.plot(
            uw_ref_res,
            z_ref / z_i_ref,
            color="black",
            label="resolved",
            linestyle="dashed",
        )

        uw_ref_sgs, z_ref, _ = read_profile_from_ref("UWSGS", case)
        ax_uw.plot(
            uw_ref_sgs,
            z_ref / z_i_ref,
            color="black",
            label="subgrid",
            linestyle="dotted",
        )

        uw_ref_tot, z_ref, _ = read_profile_from_ref("UWTOT", case)
        ax_uw.plot(uw_ref_tot, z_ref / z_i_ref, color="black", label="total")

        vw_ref_res, z_ref, _ = read_profile_from_ref("VWLE", case)
        ax_vw.plot(
            vw_ref_res, z_ref / z_i_ref, color="black", label="ref", linestyle="dashed"
        )
        vw_ref_sgs, z_ref, _ = read_profile_from_ref("VWSGS", case)
        ax_vw.plot(
            vw_ref_sgs, z_ref / z_i_ref, color="black", label="ref", linestyle="dotted"
        )

        vw_ref_tot, z_ref, _ = read_profile_from_ref("VWTOT", case)
        ax_vw.plot(vw_ref_tot, z_ref / z_i_ref, color="black", label="ref")

        w_skew_ref, z_ref, _ = read_profile_from_ref("WSKEW", case)
        ax_w_skew.plot(w_skew_ref, z_ref / z_i_ref, color="black", label="ref")

        T_ref, z_ref, label_ref = read_profile_from_ref("TXYM", case)
        ax_T.plot(T_ref, z_ref / z_i_ref, color="black", label="ref")

        Tv_ref_res, z_ref, _ = read_profile_from_ref("WTLE", case)
        ax_Tv.plot(
            Tv_ref_res, z_ref / z_i_ref, color="black", label="ref", linestyle="dashed"
        )

        Tv_ref_sgs, z_ref, _ = read_profile_from_ref("WTSGS", case)
        ax_Tv.plot(
            Tv_ref_sgs, z_ref / z_i_ref, color="black", label="ref", linestyle="dotted"
        )

        Tv_ref_tot, z_ref, _ = read_profile_from_ref("WTTOT", case)
        ax_Tv.plot(Tv_ref_tot, z_ref / z_i_ref, color="black", label="ref")

        TT_ref, z_ref, _ = read_profile_from_ref("TPS", case)
        ax_TT.plot(TT_ref, z_ref / z_i_ref, color="black", label="ref")

    z_i_list = []
    for i, (f, label) in enumerate(zip(datasets, labels)):
        y = f["y"][:]

        if normalize_BLH:
            z_i = f["y"].isel(y=(f["dtdy"][:]).argmax()).values
            print(f"{z_i = }")
            z_i_list.append(z_i)
            z_i_ref = read_BLH_from_ref(case)
        else:
            z_i = 1  # If not normalizing then just dividing by 1
            z_i_ref = 1

        # --- U and V ----------------------------------------------------------------------
        color = f"C{i}"
        ax_u.plot(f["u"][:], y / z_i, label=label, color=color)
        ax_u.set_xlabel("u, v")
        ax_u.plot(f["w"][:], y / z_i, color=color, linestyle="dashed")

        # --- uu ---------------------------------------------------------------------------
        ax_uu.plot(f["uu"][:], y / z_i, label=label, color=color)
        ax_uu.set_xlabel("uu")
        ax_uu.axvline(0, linewidth=0.5, color="black")

        # --- vv ---------------------------------------------------------------------------
        ax_vv.plot(f["ww"][:], y / z_i, label=label, color=color)
        ax_vv.set_xlabel("vv")
        ax_vv.axvline(0, linewidth=0.5, color="black")

        # --- ww ---------------------------------------------------------------------------
        ax_ww.plot(f["vv"][:], y / z_i, label=label, color=color)
        ax_ww.set_xlabel("ww")
        ax_ww.axvline(0, linewidth=0.5, color="black")

        # --- uw ---------------------------------------------------------------------------
        ax_uw.plot(
            f["uv"][:] - f["nutotdudy"][:], y / z_i, color=color
        )  # , label="total")
        ax_uw.plot(
            f["uv"][:], y / z_i, linestyle="dashed", color=color  # , label="resolved"
        )
        ax_uw.plot(
            -f["nutotdudy"][:],
            y / z_i,
            linestyle="dotted",
            color=color,
            # label="subgrid",
        )
        ax_uw.set_xlabel("uw")
        ax_uw.axvline(0, linewidth=0.5, color="black")

        # --- vw ---------------------------------------------------------------------------
        ax_vw.plot(f["vw"][:] - f["nutotdwdy"][:], y / z_i)
        ax_vw.plot(
            f["vw"][:], y / z_i, linestyle="dashed", color=color, label="resolved"
        )
        ax_vw.plot(
            -f["nutotdwdy"][:],
            y / z_i,
            linestyle="dotted",
            color=color,
            label="subgrid",
        )
        ax_vw.set_xlabel("vw")
        ax_vw.axvline(0, linewidth=0.5, color="black")

        # --- www --------------------------------------------------------------------------
        ax_nutot.plot(f["nutot"][:], y / z_i, color=color)
        ax_nutot.set_xlabel(r"Total $\nu_T$")
        ax_nutot.axvline(0, linewidth=0.5, color="black")

        # --- w skew -----------------------------------------------------------------------
        ax_w_skew.plot(f["vvv"][:] / f["vv"][:] ** (3 / 2), y / z_i, color=color)
        ax_w_skew.set_xlabel("w skewness")
        ax_w_skew.axvline(0, linewidth=0.5, color="black")

        # # --- T skew -----------------------------------------------------------------------
        # T_skew_ref, z_ref, _ = read_profile_from_ref("TSKEW")
        # ax_T_skew.plot(T_skew_ref, z_ref/z_i_ref, color="black", label="ref")
        # ax_T_skew.plot(f["vvv"][:] / f["vv"][:]**(3/2), y/z_i)
        # ax_T_skew.set_xlabel("w skewness")

        # --- theta ------------------------------------------------------------------------
        try:
            ax_T.plot(f["t"][:] - t_ref, y / z_i, color=color)
        except UnboundLocalError:
            ax_T.plot(f["t"][:], y / z_i, color=color)
        ax_T.set_xlabel(r"$\theta$")

        ax_dTdz.plot(f["dtdy"][:], y / z_i, color=color)
        ax_dTdz.set_xlabel(r"$\partial\theta/\partial z$")
        ax_dTdz.axvline(0, linewidth=0.5, color="black")

        ax_Tv.plot(f["tv"][:] - f["xitotdtdy"][:], y / z_i, label="total", color=color)
        ax_Tv.plot(
            f["tv"][:], y / z_i, linestyle="dashed", color=color, label="resolved"
        )
        ax_Tv.plot(
            -f["xitotdtdy"][:],
            y / z_i,
            linestyle="dotted",
            color=color,
            label="subgrid",
        )
        ax_Tv.set_xlabel(r"$\theta w$")
        ax_Tv.axvline(0, linewidth=0.5, color="black")

        ax_TT.plot(f["tt"][:], y / z_i, color=color)
        ax_TT.set_xlabel(r"$\theta \theta$")
        ax_TT.axvline(0, linewidth=0.5, color="black")

        ax_u.legend(frameon=False)
        ax_uw.legend(frameon=False)

    for ax in fig.axes[::3]:
        ax.set_ylabel(r"$z$ (m)")
    return fig, z_i_list


def plot_diagnostics(paths):
    """
    Plot timeseries of variables in a diagnostics.dat file

    Parameters:
        paths (str or list):    path to the case for which to plot diagnostics, or
                                list of paths to multiple cases

    Returns:
        fig:                    figure with the timeseries

    Plots timeseries of friction velocity, time step length, and CFL number for one or
    more cases.
    """
    import pandas as pd

    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    if type(paths) is not list:
        paths = [paths]
    dfs = []
    for path in paths:
        try:
            file = f"{path}/diagnostics.dat"
            df = pd.read_csv(file, names=["time", "ustar", "dt", "CFL"]).dropna()
        except FileNotFoundError:
            file = f"{path}/diagnostics.txt"
            df = pd.read_csv(file, names=["time", "ustar", "dt", "CFL"]).dropna()

        end = None
        axes[0].plot(
            df["time"][:end], df["ustar"][:end]
        )  # Friction velocity from wall model
        axes[0].set_ylabel("$u_*$")

        axes[1].plot(df["time"], df["dt"])  # Timestep length
        axes[1].set_ylabel("dt")

        axes[2].plot(df["time"], df["CFL"])
        axes[2].set_ylabel("CFL")
        # axes[2].set_ylim(0.2, None)
        axes[2].set_ylim(0.0, 0.6)
        add_hline(axes[2], 0.5)  # Add reference line at target CFL=0.5

        axes[2].set_xlabel("Time [s]")

        dfs.append(df)
    return fig, dfs


def plot_wmles_diagnostics(path: str):
    """
    Plot timeseries of variables in a wmles.dat file (surface heat flux (q), average,
    minimum and maximum Obukhov length (L))

    Parameters:
        path (str):     path to the case for which to plot diagnostics

    Returns:
        fig:            figure with the timeseries
    """
    import pandas as pd

    file = f"{path}/wmles.dat"
    df = pd.read_csv(file, names=["time", "q", "L", "L_min", "L_max"]).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
    end = None
    axes[0].plot(df["time"][:end], df["q"][:end])  # Friction velocity from wall model
    axes[0].set_ylabel("$q$")

    axes[1].plot(df["time"], df["L"], label="mean L")  # Timestep length
    axes[1].set_ylabel("$L$")

    axes[1].plot(df["time"], df["L_min"], label="min L")
    axes[1].plot(df["time"], df["L_max"], label="max L")

    axes[1].set_xlabel("Time [s]")
    axes[1].legend(frameon=False)

    return fig, df


# %%
def plot_history_points(file: str, sharex=True, sharey="col"):
    """
    Plot timeseries of variables in .his file

    Parameters:
        file (str):     name of the file with history point data
        npoints (int):  number of history points in file

    Returns:
        fig:            figure with the timeseries
        his:            array with the timeseries

    Plots timeseries of u, v, w, and theta at all points in the given .his file
    """
    npoints = int(
        np.loadtxt(
            file,
            max_rows=1,
        )
    )

    fig, axes = plt.subplots(
        npoints,
        2,
        figsize=(10, 4 * npoints),
        sharey=sharey,
        sharex=sharex,
    )

    his = np.loadtxt(
        file,
        skiprows=npoints + 1,
    )

    loc = np.loadtxt(
        file,
        skiprows=1,
        max_rows=npoints,
    )

    if npoints == 1:
        for i, label in zip(range(1, 4), ["u", "v", "w"]):
            axes[0].plot(
                his[:, 0],
                his[:, i],
                color=f"C{i-1}",
                label=label,
            )
        axes[0].set_title(f"z = {loc[1]} m", loc="left")
        axes[0].set_ylabel("m/s")
        axes[1].plot(
            his[:, 0],
            his[:, 5],
            color=f"tab:purple",
            label=r"$\theta$",
        )
        axes[1].set_ylabel("K")
        for ax in fig.axes:
            ax.legend(frameon=False)
        axes[0].set_xlabel("Time [s]")
        axes[1].set_xlabel("Time [s]")
    else:
        for j in range(npoints):
            for i, label in zip(range(1, 4), ["u", "v", "w"]):
                axes[j, 0].plot(
                    his[j::npoints, 0],
                    his[j::npoints, i],
                    color=f"C{i-1}",
                    label=label,
                )
            axes[j, 0].set_title(f"z = {loc[j, 1]} m", loc="left")
            axes[j, 0].set_ylabel("m/s")
            axes[j, 1].plot(
                his[j::npoints, 0],
                his[j::npoints, 5],
                color=f"tab:purple",
                label=r"$\theta$",
            )
            axes[j, 1].set_ylabel("K")
        for ax in fig.axes:
            ax.legend(frameon=False)
        axes[-1, 0].set_xlabel("Time [s]")
        axes[-1, 1].set_xlabel("Time [s]")

    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    return fig, his


def add_vline(ax, x: float, linestyle="dashed") -> None:
    """
    Add a thin, black vertical line to an axis

    Parameters:
        ax:             the axis on which to add the line
        x (float):      x-value at which to add the line

    Returns:
        None
    """
    ax.axvline(x, color="black", linewidth=0.5, linestyle=linestyle)


def add_hline(ax, y: float, linestyle="dashed") -> None:
    """
    Add a thin, black vertical line to an axis

    Parameters:
        ax:             the axis on which to add the line
        y (float):      y-value at which to add the line

    Returns:
        None
    """
    ax.axhline(y, color="black", linewidth=0.5, linestyle=linestyle)


# %%
def plot_logbinned_spectra(var, freq, nbins, ax, scale_by_f=False, **kwargs):
    """
    Calculate power spectrum for a single variable, and plot the log-binned average

    Parameters:
        var:        variable - array with data of which to calculate the spectrum
        freq:       sampling frequency of the data
        nbins:      number of log bins
        ax:         axis on which to plot the spectrum

    Returns:
        f:          frequency bins
        S:          spectral density
    """
    import scipy

    f, S = scipy.signal.periodogram(
        var,
        freq,
        scaling="density",
        detrend="linear",  # or "constant"
        window="boxcar",
    )
    bins = np.logspace(np.log10(np.min(f[(f > 0)])), np.log10(np.max(f)), nbins)
    binned = scipy.stats.binned_statistic(f, S, bins=bins, statistic="mean")
    mid_bins = (
        binned.bin_edges[:-1] + binned.bin_edges[1:]
    ) / 2  # The number of bins is 1 larger than the size of the spectrum, this is to plot the value in the middle of each bin
    if scale_by_f:
        ax.loglog(mid_bins, mid_bins * binned.statistic, **kwargs)
    else:
        ax.loglog(mid_bins, binned.statistic, **kwargs)
    # ax.set_xlabel("f") # frequency [Hz]
    # ax.set_ylabel("fS") # S is the spectral density, it is common to multiply it by f
    # The unit of fS is s^2/m^2 for velocity spectra, K^2 for temperature spectra
    return mid_bins, binned.statistic


def adjust_axes(axes, spines_to_remove=["top", "left"], origin=False):
    """
    Input: list of axes, which axes to remove
    Output: no output
    Removes the given spines from the axes
    """
    almost_black = "#262626"

    all_spines = ["top", "bottom", "left", "right"]

    for ax in axes:
        for spine in all_spines:
            if spine in spines_to_remove:
                ax.spines[spine].set_visible(False)
            if spine not in spines_to_remove:
                ax.spines[spine].set_linewidth(0.5)
                ax.spines[spine].set_color(almost_black)
                if spine == "left" or spine == "right":
                    ax.yaxis.set_ticks_position(spine)
                    ax.yaxis.set_label_position(spine)
                    if origin:
                        ax.spines[spine].set_position("zero")
                if spine == "top" or spine == "bottom":
                    ax.xaxis.set_ticks_position(spine)
                    ax.xaxis.set_label_position(spine)
                    if origin:
                        ax.spines[spine].set_position("zero")


def profile_broken_axis(
    variables,
    colors,
    labels,
    y,
    upper_ylim,
    lower_ylim,
    connect,
    figsize=None,
    **kwargs,
):
    """
    Create a figure plotting the vertical profile of each variable in `variables` in its
    own panel, with a broken y-axis

    Parameters:
        variables:          list of 1d xarrays with the profiles of the variables to plot
        colors:             list of colors to plot each variable in
        labels:             list pf labels for each variable
        y:                  1d xarray with vertical coordinate
        upper_ylim:         tuple with two values, giving the range of the yaxis above the break
        lower_ylim:         tuple with two values, giving the range of the yaxis below the break
        kwargs:             key word arguments to be forwarded to the plt.plot function

    Returns:
        fig:                figure with the variables plotted
        axes:               axes of the figure
    """
    upper_range = upper_ylim[1] - upper_ylim[0]
    lower_range = lower_ylim[1] - lower_ylim[0]

    ratio = lower_range / upper_range
    if not figsize:
        figsize = (4 * len(variables) + 1, 4)
    fig, axes = plt.subplots(
        2,
        len(variables),
        figsize=figsize,
        sharey="row",
        sharex="col",
        gridspec_kw={"height_ratios": [1, ratio]},
    )

    for i, (variable, color, label, connect_current) in enumerate(
        zip(variables, colors, labels, connect)
    ):
        if len(variables) > 1:
            current_axes = axes[:, i]
        else:
            current_axes = axes[:]
        broken_yaxis(
            current_axes,
            variable,
            y,
            upper_ylim,
            lower_ylim,
            color=color,
            label=label,
            **kwargs,
        )
        if connect_current:
            connect_points_between_yaxes(
                current_axes,
                variable,
                y,
                upper_ylim,
                lower_ylim,
                color,
            )

    return fig, axes


def broken_yaxis(axes, variable, y, upper_ylim, lower_ylim, color, label, **kwargs):
    """
    Plot the vertical profile of a given variable on a given axis and create a break in
    the y-axis according to the given limits

    Parameters:
        axes:               axes that will make out the top and bottom parts of the plot
        variable:           1d xarray with the profile of the variable to plot
        y:                  1d xarray with vertical coordinate
        upper_ylim:         tuple with two values, giving the range of the yaxis above the break
        lower_ylim:         tuple with two values, giving the range of the yaxis below the break
        color:              color to plot the profile in
        label:              label to put in figure legend for the variable

    Returns:
        None
    """
    for ax in axes:
        ax.plot(variable, y, color=color, label=label, **kwargs)

    axes[0].set_ylim(upper_ylim)
    axes[1].set_ylim(lower_ylim)

    axes[0].spines.bottom.set_visible(False)
    axes[1].spines.top.set_visible(False)

    axes[0].xaxis.tick_top()
    axes[0].tick_params(labeltop=False, top=False)  # Don't put tick labels at the top
    axes[1].xaxis.tick_bottom()

    d = 0.5  # Proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )

    # Create diagonal lines indicating broken axes
    axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)


def connect_points_between_yaxes(axes, variable, y, upper_ylim, lower_ylim, color):
    """
    Create a dashed line that connects the profile values across the broken y-axis

    Parameters:
        axes:               axes that will make out the top and bottom parts of the plot
        variable:           1d xarray or numpy array with the profile of the variable to plot
        y:                  1d xarray with vertical coordinate
        upper_ylim:         tuple with two values, giving the range of the yaxis above the break
        lower_ylim:         tuple with two values, giving the range of the yaxis below the break
        color:              color to plot the profile in

    Returns:
        None
    """

    from matplotlib.patches import ConnectionPatch
    import xarray as xr

    # Find which points to connect (select profile value at the top of the lower panel
    # and the bottom of the upper panel)
    if type(y) == np.ndarray:
        xyA = (variable[np.argmin(np.abs(y - lower_ylim[1]))], lower_ylim[1])
        xyB = (variable[np.argmin(np.abs(y - upper_ylim[0]))], upper_ylim[0])
    elif type(y) == xr.core.dataarray.DataArray:
        xyA = (
            variable.drop_duplicates("y").sel(y=lower_ylim[1], method="nearest"),
            lower_ylim[1],
        )
        xyB = (
            variable.drop_duplicates("y").sel(y=upper_ylim[0], method="nearest"),
            upper_ylim[0],
        )

    # Create connection
    con = ConnectionPatch(
        xyA=xyA,
        xyB=xyB,
        coordsA="data",
        coordsB="data",
        axesA=axes[1],
        axesB=axes[0],
        linestyle=(0, (5, 5)),
        linewidth=1,
        color=color,
    )

    # Add connection
    axes[0].add_artist(con)


def connect_points_between_timeseries(axes, variable, x_cutoff, color):
    """
    Create a dashed line that connects the profile values across the broken x-axis

    Parameters:
        axes:               axes that will make out the left and right parts of the plot
        variable:           1d xarray with the profile of the variable to plot
        x:                  1d xarray with horizontal coordinate
        left_xlim:          tuple with two values, giving the range of the xaxis before the break
        right_xlim:         tuple with two values, giving the range of the xaxis after the break
        color:              color to plot the profile in

    Returns:
        None
    """

    from matplotlib.patches import ConnectionPatch
    import xarray as xr

    # Find which points to connect (select profile value at the right of the left panel
    # and the left of the right panel)
    xyA = (
        x_cutoff,
        variable.sel(time=x_cutoff, method="nearest"),
    )
    xyB = (
        x_cutoff,
        variable.sel(time=x_cutoff, method="nearest"),
    )

    # Create connection
    con = ConnectionPatch(
        xyA=xyA,
        xyB=xyB,
        coordsA="data",
        coordsB="data",
        axesA=axes[0],
        axesB=axes[1],
        linestyle=(0, (5, 5)),
        linewidth=1,
        color=color,
    )

    # Add connection
    axes[1].add_artist(con)
