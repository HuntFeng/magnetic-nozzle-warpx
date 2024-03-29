import numpy as np
import scipy.constants as constants


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


def ion_sound_velocity(T_e: float, T_i: float, m_i: float, Z: int = 1):
    """
    ion sound wave velocity

    T_e: electron temperature in eV
    T_i: ion temperature in eV
    m_i: ion mass
    Z: atom number (assume Hydrogen Z=1)
    """
    T_ij = T_i / 6.242e18  # ion temperature in joules
    T_ej = T_e / 6.242e18  # electron temperature in joules
    return np.sqrt((Z * T_ej + T_ij) / m_i)


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
