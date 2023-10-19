"""Class to inject particles in a WarpX simulation based on specific source
distributions."""

import numpy as np
import util
from pywarpx import picmi, particle_containers


class FluxMaxwellian_ZInjector(object):
    """A flux Maxwellian injector class.

    Arguments:
        species (picmi.Species object): The species that will be injected.
        T (float): The temperature of the species.
        weight (float): The statistical weight for each injected particle.
        nparts (int): Number of particles to inject per timestep - each processor
            that initialized this injector will inject this many particles.
        zmin, zmax (float): Minimum and maximum z-coordinates of the flux plane.
        rmin, rmax (float): Minimum and maximum r-coordinates of the flux plane.
    """

    def __init__(
        self,
        species: picmi.Species,
        T: float,
        weight: float,
        nparts: int,
        zmin: float,
        zmax: float,
        rmin: float,
        rmax: float,
    ):
        self.species = species
        self.T = T
        self.weight = weight
        self.nparts = nparts

        self.sigma = util.thermal_velocity(self.T, self.species.m)

        self.zmin = zmin
        self.zmax = zmax
        self.rmin = rmin
        self.rmax = rmax

    def inject_parts(self):
        """Function to actually inject the simulation particles."""
        # generate random positions for each particle
        r = self.rmax * np.sqrt(np.random.uniform(self.rmin, self.rmax, self.nparts))
        theta = np.random.uniform(self.rmin, self.rmax, self.nparts) * 2 * np.pi
        x_pos = r * np.cos(theta)
        y_pos = r * np.sin(theta)
        z_pos = np.random.uniform(self.zmin, self.zmax, self.nparts)

        # sample a Gaussian for the x and y velocities
        vx_vals = np.random.normal(0, self.sigma, self.nparts)
        vy_vals = np.random.normal(0, self.sigma, self.nparts)
        # vz_vals = np.abs(np.random.normal(0, self.sigma, self.nparts))
        vz_vals = self.sigma * np.sqrt(-2.0 * np.log(1.0 - np.random.rand(self.nparts)))

        part_wrapper = particle_containers.ParticleContainerWrapper(self.species.name)
        part_wrapper.add_particles(
            x=x_pos, y=y_pos, z=z_pos, ux=vx_vals, uy=vy_vals, uz=vz_vals, w=self.weight
        )
