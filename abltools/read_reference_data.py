"""
Functions for reading vertical profiles, average BLH, and timeseries from reference
simulations performed by Peter Sullivan using the NCAR LES model.

Linnea Huusko, 2024-03-11
"""

import re
import numpy as np

ref_dir = "/cfs/klemming/projects/snic/abl-les/ABL/nec5000"


def read_profile_from_ref(variable: str, case: str):
    """
    Read vertical profile data from the reference simulation for some variable and case.

    Parameters:
        variable (str):     Name of the variable in the NCAR model
        case (str):         Name of the case ("stable", "stable_512", "free_conv",
                            "mixed", "mixed2", or "neutral")

    Returns:
        variable values:    Numpy array with the values of the variable
        z:                  Numpy array with the z values (height)
        label:              ?
    """
    if case == "stable":
        file = f"{ref_dir}/fa1.xy"
    elif case == "stable200":
        file = f"{ref_dir}/fa2.xy"
    elif case == "stable_512":
        file = f"{ref_dir}/ea4.xy"
    elif case == "free_conv":
        file = f"{ref_dir}/ra1.xy"
    elif case == "mixed":
        file = f"{ref_dir}/ra2.xy"
    elif case == "mixed2":
        file = f"{ref_dir}/ra5.xy"
    elif case == "neutral":
        file = f"{ref_dir}/ra4.xy"

    with open(file, "r") as file:
        content = file.readlines()

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
    """
    Read average BLH value from the reference simulation for some variable and case.

    Parameters:
        case (str):         Name of the case ("stable", "stable_512", "free_conv",
                            "mixed", "mixed2", or "neutral")

    Returns:
        z_i:                BLH value
    """
    if case == "stable":
        file = f"{ref_dir}/fa1.xy"
    if case == "stable_512":
        file = f"{ref_dir}/ea4.xy"
    elif case == "free_conv":
        file = f"{ref_dir}/ra1.xy"
    elif case == "mixed":
        file = f"{ref_dir}/ra2.xy"
    elif case == "mixed2":
        file = f"{ref_dir}/ra5.xy"
    elif case == "neutral":
        file = f"{ref_dir}/ra4.xy"

    with open(file, "r") as file:
        content = file.readlines()

    for l, line in enumerate(content):
        if "Zi_grad_bar" in line:
            z_i = float(line.split()[-1])

    return z_i


def read_timeseries_from_ref(variable, case):
    """
    Read timeseries data from the reference simulation for some variable and case.

    Parameters:
        variable (str):     Name of the variable in the NCAR model
        case (str):         Name of the case ("stable", "stable_512", "free_conv",
                            "mixed", "mixed2", or "neutral")

    Returns:
        variable values:    Numpy array with the values of the variable
        time:               Numpy array with the time values
    """
    if case == "stable":
        file = f"{ref_dir}/fa1.his.xy"
    if case == "stable_512":
        file = f"{ref_dir}/ea4.his.xy"
    elif case == "free_conv":
        file = f"{ref_dir}/ra1.his.xy"
    elif case == "mixed":
        file = f"{ref_dir}/ra2.his.xy"
    elif case == "mixed2":
        file = f"{ref_dir}/ra5.his.xy"
    elif case == "neutral":
        file = f"{ref_dir}/ra4.his.xy"

    with open(file, "r") as file:
        content = file.readlines()

    for i, line in enumerate(content):
        if variable in line:
            # print("Match!", i)
            I = i

    for j, line in enumerate(content[I + 1 :]):
        match = re.compile("#").match(line)

        if match:
            break

    value = []
    time = []
    for line in content[I + 1 : I + j + 1]:
        time.append(float(line.split()[0]))
        value.append(float(line.split()[1]))

    return np.array(value), np.array(time)
