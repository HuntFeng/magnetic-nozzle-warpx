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

from pywarpx import callbacks, fields, picmi
from params import Params
import numpy as np


class CurrentFreeBoundaryCondition:
    """
    A class to implement boundary conditions such that current = 0 in the nozzle.
    To use this class, we need to set upper_z boundary condition to 'dirichlet'
    """

    def __init__(
        self,
        ext: picmi.Simulation.extension,
        grid: picmi.CylindricalGrid,
        params: Params,
    ) -> None:
        self.ext = ext
        self.params = params
        # set these to None and get them later since warpx is not initialized yet
        self.phi_wrapper = None
        self.Jz_wrapper = None
        # arrays to hold potential and current density values
        self.J_arr = []
        self.phi_arr = [-6 * params.T_e]
        self.target_J = 0
        grid.potential_zmax = self.phi_arr[-1]
        print(f"STEP 0, phi_arr={self.phi_arr}")

    def upper_z(self):
        self.ext.warpx.set_potential_on_domain_boundary(
            potential_hi_z=f"{self.phi_arr[-1]}"
        )

    def lower_z(self):
        pass

    def apply_bc(self):
        self.lower_z()
        self.upper_z()

    def stats(self, Jz: np.array):
        """Calculate the mean and std of Jz along the z axis"""
        dr = self.params.dr
        Lr = self.params.Lr
        Nr, Nz = Jz.shape  # since Jz is on staggered grid, Nr = params.Nr + 1, so as Nz
        rr = np.linspace(dr / 2, dr * Nr, Nr).repeat(Nz).reshape(Nr, -1)
        J_along_z = (Jz * rr).sum(axis=0) * 2 / Lr**2  # averaged over r
        mean = np.mean(J_along_z)
        std = np.std(J_along_z)
        return mean, std

    def adjust_phi(self):
        """Adjust the potential at the nozzle exit using secant method"""
        step = self.ext.warpx.getistep(lev=0)
        if step < 50000 or step % 10000 != 0:
            return

        if self.phi_wrapper is None:
            self.phi_wrapper = fields.PhiFPWrapper()

        if self.Jz_wrapper is None:
            self.Jz_wrapper = fields.JzWrapper()

        mean, std = self.stats(self.Jz_wrapper[...])
        self.J_arr.append(mean)
        if len(self.J_arr) < 2:
            # since we use secant method, we need 2 initial guesses
            phi_new = -10 * self.params.T_e
        else:
            # use secant method to determine the next potential value
            J = self.J_arr[-1]
            J_prv = self.J_arr[-2]
            phi = self.phi_arr[-1]
            phi_prv = self.phi_arr[-2]
            if abs(J - J_prv) < 1e-3 or abs(J - self.target_J) < std / 10:
                phi_new = phi
            else:
                phi_new = phi - (J - self.target_J) * (phi - phi_prv) / (J - J_prv)
        self.phi_arr.append(phi_new)
        print(f"STEP {step}, phi_arr = {self.phi_arr}")

    def install(self):
        callbacks.installbeforeEsolve(self.apply_bc)
        # callbacks.installafterdiagnostics(self.adjust_phi) # does not work properly
        callbacks.installafterstep(self.adjust_phi)
