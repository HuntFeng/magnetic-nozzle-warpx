"""Class to inject particles in a WarpX simulation based on specific source
distributions."""

import numpy as np
import util
from pywarpx import picmi, particle_containers, libwarpx


class FluxMaxwellian_ZInjector(object):
    """A flux Maxwellian injector class.

    Arguments:
        species (picmi.Species object): The species that will be injected.
        T (float): The temperature of the species.
        weight (float): The statistical weight for each injected particle.
        nparts (int): Number of particles to inject per timestep (number of injection particles per processor is determined at runtime) 
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
        # this line has to be here since libwarpx has no amr object before initialization
        nprocs = libwarpx.amr.ParallelDescriptor.NProcs()
        nparts_per_proc = int(self.nparts / nprocs)
        # generate random positions for each particle
        r = self.rmax * np.sqrt(np.random.rand(nparts_per_proc))
        theta = 2 * np.pi * np.random.rand(nparts_per_proc)
        x_pos = r * np.cos(theta)
        y_pos = r * np.sin(theta)
        z_pos = np.random.uniform(self.zmin, self.zmax, nparts_per_proc)

        # sample a Gaussian for the x and y velocities
        # 1/sqrt(2) is to make v_perp = v_thermal
        vx_vals = np.random.normal(0, self.sigma, nparts_per_proc) / np.sqrt(2)
        vy_vals = np.random.normal(0, self.sigma, nparts_per_proc) / np.sqrt(2)
        # we want the particles to have only positive vz values
        # vz_vals = np.abs(np.random.normal(0, self.sigma, nparts_per_proc)) is not okay
        # since most of the particles will then have 0 vz
        # use the random number generator for normal dist. but drop the cos(2*pi*rand) factor
        vz_vals = self.sigma * \
            np.sqrt(-2.0 * np.log(1.0 - np.random.rand(nparts_per_proc)))

        part_wrapper = particle_containers.ParticleContainerWrapper(
            self.species.name)
        part_wrapper.add_particles(
            x=x_pos, y=y_pos, z=z_pos, ux=vx_vals, uy=vy_vals, uz=vz_vals, w=self.weight
        )
