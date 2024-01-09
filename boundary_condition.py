"""
Usage:

# In the setup, before simulation.step() and after simulation=Simulation()
bc = BoundaryCondition()
bc.install()
"""

from pywarpx import callbacks, fields
from numpy.typing import NDArray


class BoundaryCondition:
    def __init__(self) -> None:
        self._rho_wrapper = None
        self.nzguard = 2

    @property
    def rho(self) -> NDArray:
        if self.rho_wrapper is None:
            self.rho_wrapper = fields.RhoFPWrapper(0, True)
        return self._rho_wrapper[Ellipsis]

    def apply_bc(self):
        """
        Floating wall condition at the nozzle exit

        Copy the charges from second last layer to last layer and ghost cells.
        The charges are floating out of the region rather than reflected / absorbed by the wall.
        """
        self.rho[:, -self.nzguard - 1 :] = self.rho[:, -2].repeat(3).reshape(-1, 3)

    def install(self):
        callbacks.installbeforeEsolve(self.apply_bc)
