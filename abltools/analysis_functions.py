"""
Random useful functions for working with ABL simulations from Nek5000
Linnea Huusko, 2024-03-08
"""


def calculate_Ri_grad(theta0: float, dtdz, dudz, dvdz, g: float = 9.81) -> float:
    """
    Calculate gradient Richardson number

    Parameters:
        theta0 (float):             Reference temperature (potential temperature)
        dtdz:                       array with vertical derivative of the temperature
        dudz:                       array with vertical derivative of u (velocity)
        dvdz:                       array with vertical derivative of v (velocity)
        g (float):                  Gravitational acceleration

    Returns:
        Ri (float):                 Gradient Richardson number

    """

    Ri = g / theta0 * dtdz / (dudz**2 + dvdz**2)
    return Ri


def get_blh(path: str, casename: str):
    import glob
    import pymech.dataset as ds
    import xarray as xr
    from os.path import join

    datafiles = glob.glob(
        join(path, "sts" + casename + "[0-1].f[0-9][0-9][0-9][0-9][0-9]")
    )  # Filenames
    print("Reading datasets")

    # datasets = []
    z_i_list = []
    for i in datafiles:
        print(i)
        # datasets.append(ds.open_dataset(i)["s61"].rename("dtdy"))
        z_i_list.append(
            ds.open_dataset(i)["s61"].y.isel(
                y=ds.open_dataset(i)["s61"].argmax(dim="y")
            )
        )
    z_i = xr.concat(z_i_list, dim="time").sortby("time").mean(["x", "z"])
    # z_i = dtdz.y.isel(y=dtdz.argmax(dim="y"))
    return z_i


def get_diagnostics(path: str):
    import pandas as pd

    file = f"{path}/diagnostics.dat"
    df = pd.read_csv(file, names=["time", "ustar", "dt", "CFL"]).dropna()
    return df


def rolling_mean(x, n):
    import numpy as np

    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def read_variable_from_ref(variable, case):
    import re
    import numpy as np

    if case == "stable":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/stable/Sullivan/ref/fa1.xy"
    if case == "stable_512":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/stable/Sullivan/ref/ea4.xy"
    elif case == "free_conv":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/convective/Sullivan/free_conv_ref/ra1.xy"
    elif case == "mixed":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/convective/Sullivan/mixed_ref/ra2.xy"
    elif case == "mixed2":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/convective/Sullivan/mixed_ref/ra5.xy"
    elif case == "neutral":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/neutral/Sullivan/ref/ra4.xy"
    with open(file, "r") as file:
        content = file.readlines()
    # variable = "#k \"TXYM \""
    # variable = "TPS"
    for i, line in enumerate(content):
        if variable in line:
            # print("Match!", i)
            I = i

    for k, line in enumerate(content[I - 10 : I]):
        match = re.compile("#lx").match(line)
        found_label = False
        if match:
            # print("Match!", line, I-10+k)
            label = line.split('"')[1]
            found_label = True

        if found_label == False:
            label = " "
    for j, line in enumerate(content[I + 1 :]):
        match2 = re.compile("#").match(line)

        if match2:
            # print("end", j+I+1)
            break

    value = []
    z = []
    for line in content[I + 1 : I + j + 1]:
        # print(line.split())
        value.append(float(line.split()[0]))
        z.append(float(line.split()[1]))

    return np.array(value), np.array(z), label


def read_BLH_from_ref(case):
    if case == "stable":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/stable/Sullivan/ref/fa1.xy"
    elif case == "free_conv":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/convective/Sullivan/free_conv_ref/ra1.xy"
    elif case == "mixed":
        file = "/cfs/klemming/projects/snic/abl-les/ABL/convective/Sullivan/mixed_ref/ra2.xy"
    with open(file, "r") as file:
        content = file.readlines()

    for l, line in enumerate(content):
        if "Zi_grad_bar" in line:
            z_i = float(line.split()[-1])

    return z_i


def get_averages(df):
    length = df.shape[0]
    averages = {}

    for parameter, values in df.items():
        averages[parameter] = values[int(0.9 * length) :].mean()

    return averages
