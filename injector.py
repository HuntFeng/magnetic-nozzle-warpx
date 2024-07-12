"""Class to inject particles in a WarpX simulation based on specific source
distributions."""

from typing import Literal
import numpy as np
from params import Params
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
        params: Params,
        zmin: float,
        zmax: float,
        rmin: float,
        rmax: float,
        rotate: bool = False,
        dist: Literal["uniform", "parabolic"] = "uniform",
    ):
        self.species = species
        self.params = params

        self.zmin = zmin
        self.zmax = zmax
        self.rmin = rmin
        self.rmax = rmax

        # whether or not rotate injection
        self.rotate = rotate

        # whether or not to use parabolic distribution at injection
        self.dist = dist

        if self.species.name == "electrons":
            self.v_T = util.thermal_velocity(params.T_e, self.species.mass)
            self.weight = params.weight_e
            self.inject_nparts = params.inject_nparts_e
        else:
            self.v_T = util.thermal_velocity(params.T_i, self.species.mass)
            self.weight = params.weight_i
            self.inject_nparts = params.inject_nparts_i
        self.v_s = util.ion_sound_velocity(params.T_e, params.T_i, params.m_i)

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
        nparts_per_proc = int(self.inject_nparts / nprocs)
        # generate random positions for each particle
        if self.dist == "uniform":
            r = self.rmax * np.sqrt(np.random.rand(nparts_per_proc))  # uniform
        elif self.dist == "parabolic":
            r = self.rmax * np.sqrt(
                1 - np.sqrt(-np.random.rand(nparts_per_proc) + 1)
            )  # parabolic
        else:
            raise ValueError("Incorrect distribution type")
        theta = 2 * np.pi * np.random.rand(nparts_per_proc)
        x_pos = r * np.cos(theta)
        y_pos = r * np.sin(theta)
        z_pos = np.random.uniform(self.zmin, self.zmax, nparts_per_proc)

        # sample a Gaussian for the x and y velocities
        if self.rotate:
            # velocity is perpendicular to position vector
            v = 0.5 * self.v_s * self.flux_maxwellian(nparts_per_proc)
            vx_vals = -v * np.sin(theta)
            vy_vals = v * np.cos(theta)
        else:
            # velocity goes radially
            # 1/sqrt(2) is to make v_perp = v_thermal
            vx_vals = self.v_T * self.maxwellian(nparts_per_proc) / np.sqrt(2)
            vy_vals = self.v_T * self.maxwellian(nparts_per_proc) / np.sqrt(2)
        # we want the particles to have only positive vz values
        # vz_vals = np.abs(np.random.normal(0, self.sigma, nparts_per_proc)) is not okay
        # since most of the particles will then have 0 vz
        # use the random number generator for normal dist. but drop the cos(2*pi*rand) factor
        vz_vals = self.v_T * self.flux_maxwellian(nparts_per_proc)

        part_wrapper = particle_containers.ParticleContainerWrapper(self.species.name)
        part_wrapper.add_particles(
            x=x_pos, y=y_pos, z=z_pos, ux=vx_vals, uy=vy_vals, uz=vz_vals, w=self.weight
        )
