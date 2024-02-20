"""
Usage:

# In the setup, before simulation.step() and after simulation=Simulation()
bc = BoundaryCondition(params)
bc.install()
"""

from pywarpx import callbacks, fields, particle_containers
from params import Params


class CurrentFreeBoundaryCondition:
    """
    A class to implement boundary conditions such that current = 0 in the nozzle.
    To use this class, we need to set upper_z boundary condition to 'dirichlet'
    """
    def __init__(self, params: Params) -> None:
        self.params = params
        self.electrons = particle_containers.ParticleContainerWrapper("electrons")
        self.ions = particle_containers.ParticleContainerWrapper("ions")
        self.phi_wrapper = fields.PhiFPWrapper()
        self.Jz_wrapper = fields.JzWrapper()
        # arrays to hold potential and current values
        # since we use secant method, we need 2 initial guesses
        self.J_arr = []
        self.phi_arr = [0, -5*params.T_e] 
        self.target_J = params.target_J

    @property
    def phi(self):
        """A getter for potential phi"""
        return self.phi_wrapper[Ellipsis]

    @property
    def Jz(self):
        """A getter for current density in z """
        return self.Jz_wrapper[Ellipsis]
    
    def upper_z(self):
        """ Dirichlet boundary condition at the nozzle exit """
        self.J_arr.append(self.Jz[:,-1].mean())
        if (len(self.J_arr) < 2): 
            # not initialized yet
            self.phi[:,-1] = self.phi_arr[-1]
        else:
            # use secant method to determine the next potential value
            J = self.J_arr[-1]
            J_prv = self.J_arr[-2]
            phi = self.phi_arr[-1]
            phi_prv = self.phi_arr[-2]
            phi_new = phi - (J - self.target_J) * (phi - phi_prv) / (J - J_prv)
            self.phi_arr.append(phi_new)
            self.phi[:,-1] = phi_new

    def lower_z(self):
        pass

    def apply_bc(self):
        self.lower_z()
        self.upper_z()

    def install(self):
        callbacks.afterdiagnostics(self.apply_bc)
