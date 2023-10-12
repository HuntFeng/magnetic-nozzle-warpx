"""Class to inject particles in a WarpX simulation based on specific source
distributions."""

import numpy as np

# from koios import util
import util
from pywarpx import particle_containers

class FluxMaxwellian_ZInjector(object):
    """A flux Maxwellian injector class.

    Arguments:
        part_wrapper (picmi.ParticleContainerWrapper object): Used to add particles.
        species (picmi.Species object): The species that will be injected.
        T (float): The temperature of the species.
        weight (float): The statistical weight for each injected particle.
        nparts (int): Number of particles to inject per timestep - each processor
            that initialized this injector will inject this many particles.
        plane_z (float): z-coordinate of the flux plane.
        xmin, xmax (float): Minimum and maximum x-coordinates of the flux
            plane.
        ymin, ymax (float): Minimum and maximum y-coordinates of the flux
            plane
    """

    def __init__(self, species, T, weight, nparts, zmin, zmax,
                 xmin=None, xmax=None, ymin=None, ymax=None):

        #self.sim_ext = sim_ext
#         self.part_wrapper = part_wrapper
        self.species = species
        self.T = T
        self.weight = weight
        self.nparts = nparts

        self.sigma = util.thermal_velocity(self.T, self.species.m)

        self.zmin = zmin
        self.zmax = zmax

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def inject_parts(self):
        """Function to actually inject the simulation particles."""
        # generate random positions for each particle
        z_pos = np.random.uniform(self.zmin, self.zmax, self.nparts)
        x_pos = np.random.uniform(self.xmin, self.xmax, self.nparts)
        y_pos = 0.0  # np.random.uniform(self.ymin, self.ymax, self.nparts)

        # sample a Gaussian for the x and y velocities
        vx_vals = np.random.normal(0, self.sigma, self.nparts)
        vy_vals = np.random.normal(0, self.sigma, self.nparts)
        # vz_vals = np.abs(np.random.normal(0, self.sigma, self.nparts))
        vz_vals = (
            self.sigma *
            np.sqrt(-2. * np.log(1.0 - np.random.rand(self.nparts)))
        )

        #self.sim_ext.add_particles(
            #self.species.name,
#         self.part_wrapper.add_particles(
        part_wrapper = particle_containers.ParticleContainerWrapper(self.species.name)
        part_wrapper.add_particles(
            x=x_pos,
            y=y_pos,
            z=z_pos,
            ux=vx_vals,
            uy=vy_vals,
            uz=vz_vals,
            w=self.weight
        )
