"""
Usage:

# In the setup, before simulation.step() and after simulation=Simulation()
bc = BoundaryCondition(params)
bc.install()
"""

from pywarpx import callbacks, fields, particle_containers
from numpy.typing import NDArray
from params import Params


class BoundaryCondition:
    def __init__(self) -> None:
        self.rho_wrapper = None
        self.nzguard = 2

    @property
    def rho(self) -> NDArray:
        if self.rho_wrapper is None:
            self.rho_wrapper = fields.RhoFPWrapper(0, True)
        return self.rho_wrapper[Ellipsis]

    def apply_bc(self):
        """
        Floating wall condition at the nozzle exit

        Copy the charges from second last layer to last layer and ghost cells.
        The charges are floating out of the region rather than reflected / absorbed by the wall.
        """
        last = -self.nzguard - 1
        self.rho[:, last:] = self.rho[:, last - 1].repeat(3).reshape((-1, 3))

    def install(self):
        callbacks.installbeforeEsolve(self.apply_bc)


class CurrentFreeBoundaryCondition:
    def __init__(self, params: Params) -> None:
        self.electrons = particle_containers.ParticleContainerWrapper("electrons")
        self.ions = particle_containers.ParticleContainerWrapper("ions")
        self.params = params

    def upper_z(self):
        """chen_wang_etal_electric_2020
        during each time step, the electrons and ions passing the open boundary are recorded.
        Speciﬁcally, these escaping electrons are sorted according to their kinetic energy,
        then the electron with the lowest energy will be specularly reﬂected back into the domain;
        the reﬂection is carried out repeatedly until the number of escaping free electrons equals to that of escaping ions.
        """
        z_max = self.params.Lz / 2
        dz = self.params.dz
        z_i = self.ions.get_particle_arrays("z")
        z_e = self.electrons.get_particle_arrays("z")
        ux_e = self.electrons.get_particle_arrays("ux")
        uy_e = self.electrons.get_particle_arrays("uy")
        uz_e = self.electrons.get_particle_arrays("uz")
        # need to get this particles in the last cell
        e_at_boundary = (z_e >= z_max - dz) & (z_i <= z_max)
        i_at_boundary = (z_i >= z_max - dz) & (z_e <= z_max)
        # get the number of reflected electrons
        num_i_at_boundary = z_i[i_at_boundary].size()
        num_e_at_boundary = z_e[e_at_boundary].size()
        num_e_reflected = num_e_at_boundary - num_i_at_boundary
        if num_e_reflected <= 0:
            return
        # sort the kinetic energy of electrons at the boundary
        K_e = ux_e**2 + uy_e**2 + uz_e**2
        sorted_ind = K_e[e_at_boundary].argsort()
        reflected_ind = sorted_ind[:num_e_reflected]
        # reflect normal velocity
        # do nothing to the z position because the particle is still in the domain
        # in the next time step particles out of the domain will be removed by the absorbing bc
        uz_e[reflected_ind] = -uz_e[reflected_ind]

    def lower_z(self):
        """Particles removed by this boundary will be injected again through injector"""
        # do we really need to do this? the injection is already quasineutral
        # how do we maintain the quasineutrality while taking into account the removed particles?
        pass

    def apply_bc(self):
        self.lower_z()
        self.upper_z()

    def install(self):
        # call after built-in boundary conditions applied, before particles are removed from the boundary
        callbacks.installparticlescraper(self.apply_bc)
