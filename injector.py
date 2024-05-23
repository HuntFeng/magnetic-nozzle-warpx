"""Class to inject particles in a WarpX simulation based on specific source
distributions."""

import numpy as np
from params import Params
import util
from pywarpx import picmi, particle_containers, libwarpx
from scipy.stats import rv_continuous


class Parabolic(rv_continuous):
    """Parabolic distribution"""

    def _pdf(self, r):
        r0 = self.b
        return 2 / (np.pi * r0**4) * (-(r**2) + r0**2)


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
        params: Params,
        zmin: float,
        zmax: float,
        rmin: float,
        rmax: float,
        rotate: bool = False,
    ):
        self.species = species
        self.params = params

        self.zmin = zmin
        self.zmax = zmax
        self.rmin = rmin
        self.rmax = rmax

        # whether or not rotate injection
        self.rotate = rotate

        self.parabolic_dist = Parabolic(a=0, b=params.Lr)

    def flux_maxwellian(self, N: int):
        """returns N numbers sampled from flux Maxwellian distribution"""
        return np.sqrt(-2.0 * np.log(1.0 - np.random.rand(N)))

    def maxwellian(self, N: int):
        """returns N numbers sampled from Maxwellian distribution N(0,1)"""
        return np.random.normal(0, 1, N)

    def inject_parts(self):
        """Function to actually inject the simulation particles."""
        # this line has to be here since libwarpx has no amr object before initialization
        nprocs = libwarpx.amr.ParallelDescriptor.NProcs()
        params = self.params
        if self.species.name == "electrons":
            nparts_per_proc = int(params.inject_nparts_e / nprocs)
            v_T = util.thermal_velocity(params.T_e, self.species.mass)
            weight = params.weight_e
        else:
            nparts_per_proc = int(params.inject_nparts_i / nprocs)
            v_T = util.thermal_velocity(params.T_i, self.species.mass)
            weight = params.weight_i
        v_s = util.ion_sound_velocity(params.T_e, params.T_i, params.m_i)
        # generate random positions for each particle
        # r = self.rmax * np.sqrt(np.random.rand(nparts_per_proc))
        r = self.parabolic_dist.rvs(size=nparts_per_proc)
        theta = 2 * np.pi * np.random.rand(nparts_per_proc)
        x_pos = r * np.cos(theta)
        y_pos = r * np.sin(theta)
        z_pos = np.random.uniform(self.zmin, self.zmax, nparts_per_proc)

        # sample a Gaussian for the x and y velocities
        if self.rotate:
            # velocity is perpendicular to position vector
            v = 0.5 * v_s * self.flux_maxwellian(nparts_per_proc)
            vx_vals = -v * np.sin(theta)
            vy_vals = v * np.cos(theta)
        else:
            # velocity goes radially
            # 1/sqrt(2) is to make v_perp = v_thermal
            vx_vals = v_T * self.maxwellian(nparts_per_proc) / np.sqrt(2)
            vy_vals = v_T * self.maxwellian(nparts_per_proc) / np.sqrt(2)
        # we want the particles to have only positive vz values
        # vz_vals = np.abs(np.random.normal(0, self.sigma, nparts_per_proc)) is not okay
        # since most of the particles will then have 0 vz
        # use the random number generator for normal dist. but drop the cos(2*pi*rand) factor
        vz_vals = self.flux_maxwellian(nparts_per_proc)

        part_wrapper = particle_containers.ParticleContainerWrapper(self.species.name)
        part_wrapper.add_particles(
            x=x_pos, y=y_pos, z=z_pos, ux=vx_vals, uy=vy_vals, uz=vz_vals, w=weight
        )
