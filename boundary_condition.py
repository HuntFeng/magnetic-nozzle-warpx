"""
Usage:

# If compute backend is GPU, then we need to set warpx_amrex_the_arena_is_managed=True

simulation = picmi.Simulation(verbose=1, warpx_amrex_the_arena_is_managed=True)

this allows us to access and manipulate data in CPU.

# Install the boundary condition in the simulation setup

def simulation_setup():
    ...
    bc = BoundaryCondition(params)
    bc.install()
    ...
"""

from pywarpx import callbacks, fields
from params import Params
import numpy as np


class CurrentFreeBoundaryCondition:
    """
    A class to implement boundary conditions such that current = 0 in the nozzle.
    To use this class, we need to set upper_z boundary condition to 'dirichlet'
    """

    def __init__(self, sim_ext, params: Params) -> None:
        self.sim_ext = sim_ext
        self.params = params
        # set these to None and get them later since warpx is not initialized yet
        self.phi_wrapper = None
        self.Jz_wrapper = None
        # arrays to hold potential and current density values
        self.J_arr = []
        self.phi_arr = [-5 * params.T_e]
        # self.target_J = params.target_J
        self.target_J = 0

    def upper_z(self):
        if self.phi_wrapper is None:
            self.phi_wrapper = fields.PhiFPWrapper()
        self.phi_wrapper[:, -1] = self.phi_arr[-1]

    def lower_z(self):
        pass

    def apply_bc(self):
        self.lower_z()
        self.upper_z()

    def mean(self, Jz_at_exit: np.array):
        """Calculate average J at the nozzle exit
        Jz_at_exit: Jz values along radial direction, shape=(Nr+1,1)
        """
        dr = self.params.dr
        Lr = self.params.Lr
        N = Jz_at_exit.shape[0]  # N = Nr + 1
        r_grid = np.linspace(dr / 2, dr * N, N)
        return (r_grid * Jz_at_exit).sum() * 2 / Lr**2

    def adjust_phi(self):
        """Adjust the potential at the nozzle exit using secant method"""
        if not self.sim_ext.warpx.getistep(lev=0) < 800000:
            return

        if self.phi_wrapper is None:
            self.phi_wrapper = fields.PhiFPWrapper()

        if self.Jz_wrapper is None:
            self.Jz_wrapper = fields.JzWrapper()

        self.J_arr.append(self.mean(self.Jz_wrapper[:, 0]))
        if len(self.J_arr) < 2:
            # since we use secant method, we need 2 initial guesses
            self.phi_arr.append(-10 * self.params.T_e)
        else:
            # use secant method to determine the next potential value
            J = self.J_arr[-1]
            J_prv = self.J_arr[-2]
            phi = self.phi_arr[-1]
            phi_prv = self.phi_arr[-2]
            if abs(J - J_prv) < 1e-3:
                phi_new = phi
            else:
                phi_new = phi - (J - self.target_J) * (phi - phi_prv) / (J - J_prv)
            self.phi_arr.append(phi_new)
        self.phi_wrapper[:, -1] = self.phi_arr[-1]

    def install(self):
        callbacks.installbeforeEsolve(self.apply_bc)
        callbacks.installafterdiagnostics(self.adjust_phi)
