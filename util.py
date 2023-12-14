import numpy as np
import scipy as sp
import re

# constants
constants = sp.constants


def inertial_length(n: float):
    """electron inertial length in m"""
    return constants.c / plasma_freq(n)


def J_to_eV(energy: float):
    return energy * 6.242e18


def plasma_freq(n: float):
    """
    cold electron plasma frequency in Hz

    n0: number density
    """
    return np.sqrt(n * constants.e**2 / (constants.m_e * constants.epsilon_0))


def cyclotron_freq(m: float, B: float):
    """
    cyclotron frequency of particle with unit charge in Hz

    m: mass of particle in kg
    B: magnetic field in T
    """
    return constants.e * B / m


def thermal_velocity(T: float, m: float):
    """
    thermal velocity of a species in 3d, unit: m/s

    using the definition of "the most probable speed"
    https://en.wikipedia.org/wiki/Thermal_velocity

    T: temperature in eV
    m: particle mass in kg
    """
    T_j = T / 6.242e18  # temperature in joules
    return np.sqrt(2 * T_j / m)


def debye_length(T: float, n: float):
    """
    Debye length in meter

    T: particle temperature in eV
    n0: particle number density
    """
    return 7430 * np.sqrt(T / n)


def larmor_radius(m: float, v_perp: float, q: float, B: float):
    """
    Larmor radius of a particle in meter

    m: mass in kg
    v_perp: v_perp in m/s
    q: charge in C
    B: magnetic field strength in T
    """
    return m * v_perp / q / B


"""Postprocessing functions"""


def extract_data(filename):
    """
    Extarct the time steps from a standard warpx output log

    each column corresponds to the 6 numbers in the output log
    time_date[:,0]: STEP
    time_date[:,1]: TIME
    time_date[:,2]: DT
    time_date[:,3]: Evolve time
    time_date[:,4]: This step
    time_date[:,5]: Avg. per step
    """
    regex_core = re.compile(r"MPI initialized with ([0-9]*) MPI processes")
    regex_omp = re.compile(r"OMP initialized with ([0-9]*) OMP threads")
    regex_step = re.compile(
        r"STEP [0-9]* ends.*\n.* Avg\. per step = ([0-9]*[.])?[0-9]+ s", re.MULTILINE
    )
    regex_mlmg = re.compile(
        r"MLMG: Final Iter.*\n.* Bottom = ([0-9]*[.])?[0-9]+", re.MULTILINE
    )
    regex_real = re.compile(r" -?[\d.]+(?:e-?\d+)?", re.MULTILINE)
    specs = {}

    print("Processing " + filename + " ...", end="")
    with open(filename) as f:
        text = f.read()
        specs["cores"] = int(regex_core.search(text).group(1))
        specs["omp_threads"] = int(regex_omp.search(text).group(1))
        step_strings = [s.group(0) for s in regex_step.finditer(text)]
        mlmg_strings = [s.group(0) for s in regex_mlmg.finditer(text)]

    time_data = np.zeros([len(step_strings), 6])
    mlmg_data = np.zeros([len(step_strings), 6])
    for i, ss in enumerate(step_strings):
        numbers = regex_real.findall(ss)
        time_data[i, :] = np.array(numbers)
    for i, ss in enumerate(mlmg_strings):
        numbers = regex_real.findall(ss)
        mlmg_data[i, :] = np.array(numbers)

    print("...done!")
    return specs, time_data, mlmg_data
