"""2D, fully kinetic simulation of a magnetic mirror. This input is modelled
after the work of Wetherton et al. 2021.
Written in Oct 2022 by Roelof Groenewald.
Edited in Sep 2023 by Hunt Feng, using WarpX 23.11. 
"""
import os
import shutil
import numpy as np
from mpi4py import MPI as mpi
from pywarpx import callbacks, picmi
from datetime import datetime

import util
import magnetic_field
import injector
from params import Params

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(verbose=1)

#######################################################################
# Begin physical parameters                                           #
#######################################################################
params = Params()
# domain size in m
params.Lr = 0.01  # has to be smaller than the coil radius
params.Lz = 0.10

# spatial resolution in number of cells
params.Nr = 256
params.Nz = 2048
# params.Nr = 16
# params.Nz = 32

# mirror ratio
# R and Bmax determine the coil radius
params.R = 2.0
params.B_max = 0.30  # T

# use a reduced ion mass for faster simulations
params.m_e = util.constants.m_e
params.m_i = 400.0 * util.constants.m_e

# we set the voltage on the divertor side (in units of Te) since that is
# ultimately what we want to look at
params.V_divertor = -2.5

# initial seed macroparticle density
params.nppc_seed = 5  # 800

# total simulation time in ion thermal crossing times
params.crossing_times = 0.5


#######################################################################
# End global user parameters and user input                           #
#######################################################################


class MagneticMirror2D(object):
    def __init__(self):
        # initial electron density m^{-3}
        params.n0 = 1e16

        # temperature
        params.T_e = 300
        params.T_i = 1

        # domain size, unit: m
        params.dr = params.Lr / params.Nr
        params.dz = params.Lz / params.Nz

        # the given mirror ratio can only be achieved with the given Lz and
        # B_max for a unique coil radius
        params.R_coil = 0.5 * params.Lz / np.sqrt(params.R ** (2.0 / 3.0) - 1.0)
        self.coil = magnetic_field.CoilBField(R=params.R_coil, B_max=params.B_max)
        # self.coil.plot_field(params.Lr/2.0/params.R_coil, params.Lz/2.0/params.R_coil)

        # simulation timestep
        # params.dt = 0.07 / util.plasma_freq(params.n0)
        # params.dt = params.dz / (
        #     5.0 * util.thermal_velocity(params.T_e, params.m_e)
        # )
        params.dt = 0.5 / util.cyclotron_freq(params.m_e, params.B_max)

        # calculate the ion crossing time to get the total simulation time
        params.ion_crossing_time = params.Lz / util.thermal_velocity(
            params.T_i, params.m_i
        )
        params.total_steps = int(
            np.ceil(params.crossing_times * params.ion_crossing_time / params.dt)
        )
        params.diag_steps = int(params.total_steps / 100)

        # for debug use
        params.total_steps = 2000
        params.diag_steps = 100

        # calculate the flux from the thermal plasma reservoir
        params.flux_e = (
            params.n0
            * util.thermal_velocity(params.T_e, util.constants.m_e)
            / np.sqrt(2.0 * np.pi)
        )
        params.flux_i = params.flux_e * np.sqrt(util.constants.m_e / params.m_i)
        # params.flux_i = params.n0 * util.thermal_velocity(params.T_i, params.m_i)

        # check spatial resolution
        params.debye_length = util.debye_length(params.T_e, params.n0)

        self.simulation_setup()

    def simulation_setup(self):
        #######################################################################
        # Set geometry, boundary conditions and timestep                      #
        #######################################################################

        self.grid = picmi.CylindricalGrid(
            number_of_cells=[params.Nr, params.Nz],
            warpx_max_grid_size_x=8,  # max num_cells in a grid in r direction
            warpx_max_grid_size_y=32,  # max num_cells in a grid in z direction
            warpx_blocking_factor_x=4,  # min num_cells in a grid in r direction
            warpx_blocking_factor_y=16,  # min num_cells in a grid in z direction
            lower_bound=[0, -params.Lz / 2.0],
            upper_bound=[params.Lr, params.Lz / 2.0],
            lower_boundary_conditions=["none", "dirichlet"],
            upper_boundary_conditions=["neumann", "neumann"],
            lower_boundary_conditions_particles=["reflecting", "absorbing"],
            upper_boundary_conditions_particles=["absorbing", "absorbing"],
        )
        simulation.time_step_size = params.dt
        simulation.max_steps = params.total_steps
        simulation.load_balance_intervals = 50

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
            # initial_distribution=picmi.UniformDistribution(
            #     density=params.n0,
            #     rms_velocity=[util.thermal_velocity(params.T_e, util.constants.m_e)] * 3,
            # ),
        )
        self.electrons.m = params.m_e

        self.ions = picmi.Species(
            particle_type="H",
            name="ions",
            charge_state=1,
            mass=params.m_i,
            # warpx_save_particles_at_zlo=True,
            # warpx_save_particles_at_zhi=True,
            # initial_distribution=picmi.UniformDistribution(
            #     density=params.n0,
            #     rms_velocity=[util.thermal_velocity(params.T_i, params.m_i)] * 3,
            # ),
        )
        self.ions.m = params.m_i

        layout = picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=params.nppc_seed, grid=self.grid
        )

        simulation.add_species(self.electrons, layout=layout)
        simulation.add_species(self.ions, layout=layout)

        #######################################################################
        # Particle injection                                                  #
        #######################################################################

        params.inject_nparts_e = 4000
        nparts_e = params.inject_nparts_e // comm.size
        weight_e = params.flux_e * params.dt * params.Lr / (nparts_e * comm.size)
        self.electron_injector = injector.FluxMaxwellian_ZInjector(
            species=self.electrons,
            T=params.T_e,
            weight=weight_e,
            nparts=nparts_e,
            zmin=-params.Lz / 2.0 + 1.0 * params.dz,
            zmax=-params.Lz / 2.0 + 2.0 * params.dz,
            rmin=0,
            rmax=params.Lr / 2,
        )

        params.inject_nparts_i = params.inject_nparts_e
        nparts_i = nparts_e
        weight_i = params.flux_i * params.dt * params.Lr / (nparts_i * comm.size)
        self.ion_injector = injector.FluxMaxwellian_ZInjector(
            species=self.ions,
            T=params.T_i,
            weight=weight_i,
            nparts=nparts_i,
            zmin=-params.Lz / 2.0 + 1.0 * params.dz,
            zmax=-params.Lz / 2.0 + 2.0 * params.dz,
            rmin=0,
            rmax=params.Lr / 2,
        )

        callbacks.installparticleinjection(self.electron_injector.inject_parts)
        callbacks.installparticleinjection(self.ion_injector.inject_parts)

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        self.field_diag = picmi.FieldDiagnostic(
            name="diag",
            grid=self.grid,
            period=params.diag_steps,
            data_list=["phi", "rho_electrons", "rho_ions", "part_per_cell"],
            warpx_dump_rz_modes=True,
            write_dir=diags_dirname,
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        self.particle_diag = picmi.ParticleDiagnostic(
            name="diag",
            period=params.diag_steps,
            species=[self.ions, self.electrons],
            data_list=["momentum", "weighting"],
            write_dir=diags_dirname,
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(self.field_diag)
        simulation.add_diagnostic(self.particle_diag)

        #######################################################################
        # Initialize run and print diagnostic info                            #
        #######################################################################
        simulation.initialize_inputs()
        simulation.initialize_warpx()

    def run_sim(self):
        if comm.Get_rank() == 0:
            params.save(f"{diags_dirname}/params.json")
        simulation.write_input_file(file_name=f"{diags_dirname}/wrapx_used_inputs")
        simulation.step(params.total_steps)


def create_diags_dir():
    if comm.Get_rank() == 0:
        if os.path.exists(diags_dirname):
            shutil.rmtree(diags_dirname)
        os.mkdir(diags_dirname)


if __name__ == "__main__":
    # diagnostics dir (set name in rank 0 then broadcast it to ensure the name is the same in everyrank)
    diags_dirname = (
        f"diags{datetime.now().strftime('%Y%m%d%H%M')}" if comm.Get_rank() == 0 else ""
    )
    diags_dirname = comm.bcast(diags_dirname, root=0)
    create_diags_dir()
    my_2d_mirror = MagneticMirror2D()
    my_2d_mirror.run_sim()
