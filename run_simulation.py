"""2D, fully kinetic simulation of a magnetic mirror. This input is modelled
after the work of Wetherton et al. 2021.
Written in Oct 2022 by Roelof Groenewald.
Edited in Sep 2023 by Hunt Feng, using warpx 23.09 and its pywarpx wheel. 
"""
import os
import shutil
import time
import numpy as np
from mpi4py import MPI as mpi
from pywarpx import callbacks, fields, particle_containers, picmi

import util
import magnetic_field
import injector
from params import Params

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(verbose=0)
# make a shorthand for simulation.extension since we use it a lot
sim_ext = simulation.extension

#######################################################################
# Begin physical parameters                                           #
#######################################################################
params = Params()
# domain size in m
# params.Lr = 0.03
# params.Lz = 0.10
params.Lr = 3e-5
params.Lz = 1e-4

# spatial resolution in number of cells
# params.Nr = int(2**13)
# params.Nz = int(2**15)
params.Nr = int(2**4)
params.Nz = int(2**5)

# mirror ratio
params.R = 5.0
params.B_max = 30.0  # T

# desired electron beta in mirror throat
params.beta = 0.03

# desired electron plasma to cyclotron frequency ratio
params.w_pe_to_w_ce = 1.22

# use a reduced ion mass for faster simulations
params.m_ion = 400.0 * util.constants.m_e
# m_ion = util.constants.m_p

# we set the voltage on the divertor side (in units of Te) since that is
# ultimately what we want to look at
params.V_divertor = -2.5

# initial seed macroparticle density
params.nppc_seed = 5  # 800

# total simulation time in ion thermal crossing times
params.crossing_times = 8

#######################################################################
# End global user parameters and user input                           #
#######################################################################


class MagneticMirror2D(object):
    def __init__(self):
        # the initial electron density is calculated to give the desired ratio
        # of plasma to cyclotron frequency
        params.n0 = (params.w_pe_to_w_ce * params.B_max / util.constants.c) ** 2 / (
            util.constants.mu_0 * util.constants.m_e
        )

        # calculate the electron temperature to get the desired beta in the
        # mirror throat
        # p=nT (T in unit of joule), and p_mag = B^2/2\mu_0
        params.Te = util.J_to_eV(
            params.beta * params.B_max**2 / (2.0 * util.constants.mu_0 * params.n0)
        )
        # electron and ion temperatures are taken to be equal
        params.Ti = params.Te

        # calculate the electron inertial length
        params.d_e = util.inertial_length(params.n0)

        # calculate the domain size in terms of electron inertial length, unit: m
        # params.Lr = params.Lr * params.d_e
        # params.Lz = params.Lz * params.d_e

        params.dr = params.Lr / params.Nr
        params.dz = params.Lz / params.Nz

        # the given mirror ratio can only be achieved with the given Lz and
        # B_max for a unique coil radius
        params.R_coil = 0.5 * params.Lz / np.sqrt(params.R ** (2.0 / 3.0) - 1.0)
        self.coil = magnetic_field.CoilBField(R=params.R_coil, B_max=params.B_max)
        # self.coil.plot_field(params.Lr/2.0/params.R_coil, params.Lz/2.0/params.R_coil)

        # calculate electron plasma frequency
        params.w_pe = util.plasma_freq(params.n0)

        # params.dt = 0.07 / w_pe
        # simulation timestep from electron CFL
        params.dt = params.dz / (
            5.0 * util.thermal_velocity(params.Te, util.constants.m_e)
        )

        # calculate the ion crossing time to get the total simulation time
        params.ion_crossing_time = params.Lz / util.thermal_velocity(
            params.Ti, params.m_ion
        )
        params.total_steps = int(
            np.ceil(params.crossing_times * params.ion_crossing_time / params.dt)
        )
        params.diag_steps = int(params.total_steps / 20.0)

        # for debug use
        params.total_steps = 100
        params.diag_steps = 10

        # calculate the flux from the thermal plasma reservoir
        params.flux_e = (
            params.n0
            * util.thermal_velocity(params.Te, util.constants.m_e)
            / np.sqrt(2.0 * np.pi)
        )
        params.flux_i = params.flux_e * np.sqrt(util.constants.m_e / params.m_ion)
        # params.flux_i = (
        #     params.n0 * util.thermal_velocity(params.Ti, params.m_ion)
        # )

        # sanity checks
        # check spatial resolution
        params.debye_length = util.Debye_length(params.Te, params.n0)
        # assert params.dz < debye_length

        # assert np.isclose(params.coil.get_B_field(0, 0)[1], params.B_max)
        # assert np.isclose(
        #      params.coil.get_B_field(0, 0)[1] / params.coil.get_B_field(0, params.Lz/2.0)[1],
        #     params.R
        # )
        # B_min = params.B_max/params.R
        # assert np.isclose(params.coil.get_B_field(0, params.Lz/2.0)[1], B_min)
        # w_ce = util.cyclotron_freq(util.constants.m_e, params.B_max)
        # assert np.isclose(w_pe / w_ce, params.w_pe_to_w_ce)

        if comm.rank == 0:
            print("Starting simulation with parameters:\n", params)
        self.simulation_setup()

    def simulation_setup(self):
        #######################################################################
        # Set geometry, boundary conditions and timestep                      #
        #######################################################################

        self.grid = picmi.CylindricalGrid(
            number_of_cells=[params.Nr, params.Nz],
            warpx_max_grid_size=params.Nz,
            lower_bound=[0, -params.Lz / 2.0],
            upper_bound=[params.Lr, params.Lz / 2.0],
            lower_boundary_conditions=["none", "dirichlet"],
            upper_boundary_conditions=["neumann", "dirichlet"],
            lower_boundary_conditions_particles=["reflecting", "absorbing"],
            upper_boundary_conditions_particles=["absorbing", "absorbing"],
        )
        simulation.time_step_size = params.dt
        simulation.max_steps = params.total_steps
        simulation.load_balance_intervals = params.total_steps // 100

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        self.solver = picmi.ElectrostaticSolver(
            grid=self.grid,
            method="Multigrid",
            required_precision=1e-6,
            # higher the number, more verbose it is (default 2)
            warpx_self_fields_verbosity=0,
        )
        simulation.solver = self.solver

        simulation.applied_fields = [
            picmi.AnalyticAppliedField(
                Bx_expression=self.coil.get_Bx_expression(),
                By_expression=0.0,
                Bz_expression=self.coil.get_Bz_expression(),
            )
        ]

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            # warpx_save_particles_at_zlo=True,
            # warpx_save_particles_at_zhi=True,
            initial_distribution=picmi.UniformDistribution(
                density=params.n0,
                rms_velocity=[util.thermal_velocity(params.Te, util.constants.m_e)] * 3,
            ),
        )
        self.electrons.m = util.constants.m_e

        self.ions = picmi.Species(
            particle_type="H",
            name="ions",
            charge_state=1,
            mass=params.m_ion,
            # warpx_save_particles_at_zlo=True,
            # warpx_save_particles_at_zhi=True,
            initial_distribution=picmi.UniformDistribution(
                density=params.n0,
                rms_velocity=[util.thermal_velocity(params.Ti, params.m_ion)] * 3,
            ),
        )
        self.ions.m = params.m_ion

        layout = picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=params.nppc_seed, grid=self.grid
        )

        simulation.add_species(self.electrons, layout=layout)
        simulation.add_species(self.ions, layout=layout)

        #######################################################################
        # Particle injection                                                  #
        #######################################################################

        # nparts_e = 400000 // comm.size
        nparts_e = 4000 // comm.size
        weight_e = params.flux_e * params.dt * params.Lr / (nparts_e * comm.size)
        self.electron_injector = injector.FluxMaxwellian_ZInjector(
            species=self.electrons,
            T=params.Te,
            weight=weight_e,
            nparts=nparts_e,
            # plane_z=-params.Lz / 2.0 + 1e9,
            zmin=-params.Lz / 3.0 + 1.5 * params.dz,
            zmax=-params.Lz / 2.0 + 3.0 * params.dz,
            # xmin=-params.Lr / 2.0,
            # xmax=params.Lr / 2.0,
            # ymin=-1e-6,
            # ymax=1e-6,
            rmin=0,
            rmax=params.Lr / 2,
        )

        nparts_i = nparts_e
        weight_i = params.flux_i * params.dt * params.Lr / (nparts_i * comm.size)

        self.ion_injector = injector.FluxMaxwellian_ZInjector(
            species=self.ions,
            T=params.Ti,
            weight=weight_i,
            nparts=nparts_i,
            zmin=-params.Lz / 2.0 + 1.5 * params.dz,
            zmax=-params.Lz / 2.0 + 3.0 * params.dz,
            # xmin=-params.Lr / 2.0,
            # xmax=params.Lr / 2.0,
            # ymin=-1e-6,
            # ymax=1e-6,
            rmin=0,
            rmax=params.Lr / 2,
        )

        callbacks.installparticleinjection(self.electron_injector.inject_parts)
        callbacks.installparticleinjection(self.ion_injector.inject_parts)

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        callbacks.installafterinit(self._create_diags_dir)
        callbacks.installafterstep(self.text_diag)
        callbacks.installafterstep(self.phi_diag)
        callbacks.installafterstep(self.rho_diag)

        #######################################################################
        # Initialize run and print diagnostic info                            #
        #######################################################################

        simulation.initialize_inputs()
        simulation.initialize_warpx()

    def _create_diags_dir(self):
        if sim_ext.getMyProc() == 0:
            if os.path.exists("diags"):
                shutil.rmtree("diags")
            os.mkdir("diags")
            os.mkdir("diags/phi")
            os.mkdir("diags/rho")

        params.save("diags/params.json")
        self.phi_wrapper = fields.PhiFPWrapper(0, False)
        self.rho_wrapper = fields.RhoFPWrapper(0, False)

    def text_diag(self):
        """Diagnostic function to print out timing data and particle numbers."""
        step = sim_ext.getistep(0)
        if step % params.diag_steps != 0:
            return

        wall_time = time.time() - self.prev_time
        steps = step - self.prev_step
        step_rate = steps / wall_time

        self.electron_wrapper = particle_containers.ParticleContainerWrapper(
            "electrons"
        )
        self.ion_wrapper = particle_containers.ParticleContainerWrapper("ions")
        status_dict = {
            "step": step,
            "nplive_electrons": self.electron_wrapper.get_particle_count(False),
            "nplive_ions": self.ion_wrapper.get_particle_count(False),
            "wall_time": wall_time,
            "step_rate": step_rate,
            "diag_steps": params.diag_steps,
            "iproc": None,
        }

        diag_string = (
            "Step #{step:6d}; "
            "{nplive_electrons} electrons; "
            "{nplive_ions} ions; "
            "{wall_time:6.1f} s wall time; "
            "{step_rate:4.2f} steps/s"
        )

        if sim_ext.getMyProc() == 0:
            print(diag_string.format(**status_dict))

        self.prev_time = time.time()
        self.prev_step = step

    def phi_diag(self):
        step = sim_ext.getistep(0)
        if step % params.diag_steps != 0:
            return

        data = self.phi_wrapper[...] / params.Te

        if sim_ext.getMyProc() != 0:
            return

        np.save(f"diags/phi/phi_{step:08d}.npy", data)

    def rho_diag(self):
        """Charge density"""
        step = sim_ext.getistep(0)
        if step % params.diag_steps != 0:
            return

        # sim_ext.depositChargeDensity("ions", 0)
        # data = self.rho_wrapper[...][:, :, 0] / util.constants.e / params.n0
        # sim_ext.depositChargeDensity("ions", 0)
        # sim_ext.depositChargeDensity("electrons", 0)
        data = self.rho_wrapper[...][:, :] / util.constants.e / params.n0

        if sim_ext.getMyProc() != 0:
            return

        np.save(f"diags/rho/rho_{step:08d}.npy", data)

    def run_sim(self):
        self.prev_time = time.time()
        self.start_time = self.prev_time
        self.prev_step = 0

        simulation.step(params.total_steps)


my_2d_mirror = MagneticMirror2D()
my_2d_mirror.run_sim()
