"""2D, fully kinetic simulation of a magnetic mirror. This input is modelled
after the work of Wetherton et al. 2021.
Written in Oct 2022 by Roelof Groenewald.
Edited in Sep 2023 by Hunt Feng, using WarpX 23.11. 
"""

import argparse
import numpy as np
from pywarpx import callbacks, picmi

from boundary_condition import CurrentFreeBoundaryCondition

import util
import magnetic_field
import injector
from params import Params

# warpx_amrex_the_arena_is_managed=True to access fields data in GPU
# https://github.com/ECP-WarpX/WarpX/issues/4715
simulation = picmi.Simulation(verbose=1, warpx_amrex_the_arena_is_managed=True)

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

# domain size, unit: m
params.dr = params.Lr / params.Nr
params.dz = params.Lz / params.Nz

# mirror ratio
# R and Bmax determine the coil radius
params.R = 2.0
params.B_max = 1.0  # T

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
        params.T_e = 100
        params.T_i = 1

        params.v_Te = util.thermal_velocity(params.T_e, params.m_e)
        params.v_Ti = util.thermal_velocity(params.T_i, params.m_i)

        # the given mirror ratio can only be achieved with the given Lz and
        # B_max for a unique coil radius
        params.R_coil = 0.5 * params.Lz / np.sqrt(params.R ** (2.0 / 3.0) - 1.0)
        self.coil = magnetic_field.CoilBField(R=params.R_coil, B_max=params.B_max)
        # self.coil.plot_field(params.Lr/2.0/params.R_coil, params.Lz/2.0/params.R_coil)

        # simulation timestep
        # params.dt = 0.07 / util.plasma_freq(params.n0)
        params.dt = params.dz / (5.0 * params.v_Te)
        # params.dt = 0.5 / util.cyclotron_freq(params.m_e, params.B_max)

        # calculate the ion crossing time to get the total simulation time
        params.ion_crossing_time = params.Lz / params.v_Ti
        params.total_steps = int(
            np.ceil(params.crossing_times * params.ion_crossing_time / params.dt)
        )
        params.diag_steps = int(params.total_steps / 100)

        # for debug use
        params.total_steps = 100000
        params.diag_steps = 1000
        # params.total_steps = 10
        # params.diag_steps = 5

        # calculate the flux from the thermal plasma reservoir
        params.flux_e = params.n0 * params.v_Te
        # make the injection currect quasineutral
        # params.flux_i = params.flux_e
        # by setting less ion flux, less electrons will be reflected to the entrance
        params.flux_i = params.flux_e * (params.m_e / params.m_i) ** 0.5

        # check spatial resolution
        params.debye_length = util.debye_length(params.T_e, params.n0)

        self.simulation_setup()

    def simulation_setup(self):
        #######################################################################
        # Set geometry, boundary conditions and timestep                      #
        #######################################################################
        warpx_max_grid_size_x = 8 if args.cpu else 256
        warpx_max_grid_size_y = 32 if args.cpu else 256
        warpx_blocking_factor_x = 4 if args.cpu else 128
        warpx_blocking_factor_y = 16 if args.cpu else 128
        grid = picmi.CylindricalGrid(
            number_of_cells=[params.Nr, params.Nz],
            warpx_max_grid_size_x=warpx_max_grid_size_x,  # max num_cells in a grid in r direction
            warpx_max_grid_size_y=warpx_max_grid_size_y,  # max num_cells in a grid in z direction
            warpx_blocking_factor_x=warpx_blocking_factor_x,  # min num_cells in a grid in r direction
            warpx_blocking_factor_y=warpx_blocking_factor_y,  # min num_cells in a grid in z direction
            lower_bound=[0, -params.Lz / 2.0],
            upper_bound=[params.Lr, params.Lz / 2.0],
            lower_boundary_conditions=["none", "dirichlet"],
            upper_boundary_conditions=["neumann", "dirichlet"],
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
            grid=grid,
            method="Multigrid",
            required_precision=1e-6,
            # higher the number, more verbose it is (default 2)
            warpx_self_fields_verbosity=0,
        )
        simulation.solver = self.solver

        simulation.applied_fields = [
            picmi.AnalyticAppliedField(
                Bx_expression=self.coil.get_Bx_expression(),
                By_expression=self.coil.get_By_expression(),
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
            n_macroparticles_per_cell=params.nppc_seed, grid=grid
        )

        simulation.add_species(self.electrons, layout=layout)
        simulation.add_species(self.ions, layout=layout)

        #######################################################################
        # Boundary condition
        #######################################################################
        bc = CurrentFreeBoundaryCondition(simulation.extension, grid, params)
        bc.install()

        #######################################################################
        # Particle injection                                                  #
        #######################################################################

        # params.inject_nparts_e = 4000
        params.inject_nparts_e = 400
        r_inject = params.Lr / 2
        area_inject = util.constants.pi * r_inject**2
        weight_e = params.flux_e * params.dt * area_inject / params.inject_nparts_e
        self.electron_injector = injector.FluxMaxwellian_ZInjector(
            species=self.electrons,
            T=params.T_e,
            weight=weight_e,
            nparts=params.inject_nparts_e,
            zmin=-params.Lz / 2.0 + 1.0 * params.dz,
            zmax=-params.Lz / 2.0 + 2.0 * params.dz,
            rmin=0,
            rmax=r_inject,
        )

        params.inject_nparts_i = params.inject_nparts_e
        weight_i = params.flux_i * params.dt * area_inject / params.inject_nparts_i
        self.ion_injector = injector.FluxMaxwellian_ZInjector(
            species=self.ions,
            T=params.T_i,
            weight=weight_i,
            nparts=params.inject_nparts_i,
            zmin=-params.Lz / 2.0 + 1.0 * params.dz,
            zmax=-params.Lz / 2.0 + 2.0 * params.dz,
            rmin=0,
            rmax=r_inject,
        )

        callbacks.installparticleinjection(self.electron_injector.inject_parts)
        callbacks.installparticleinjection(self.ion_injector.inject_parts)

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        self.field_diag = picmi.FieldDiagnostic(
            name="diag",
            grid=grid,
            period=params.diag_steps,
            data_list=["phi", "rho_electrons", "rho_ions", "part_per_cell", "J", "E"],
            warpx_dump_rz_modes=True,
            write_dir=args.outdir,
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(self.field_diag)

        # self.particle_diag = picmi.ParticleDiagnostic(
        #     name="diag",
        #     period=params.diag_steps,
        #     species=[self.ions, self.electrons],
        #     data_list=["position", "momentum", "weighting"],
        #     write_dir=args.outdir,
        #     warpx_format="openpmd",
        #     warpx_openpmd_backend="h5",
        # )
        # simulation.add_diagnostic(self.particle_diag)

        self.checkpoint = picmi.Checkpoint(
            name="checkpoint",
            period=params.total_steps // 4,
            write_dir=args.outdir,
            warpx_file_prefix="checkpoint",
        )
        simulation.add_diagnostic(self.checkpoint)

    def run_sim(self):
        """Initialize run and print diagnostic info"""
        if args.restart is None:
            # if libwarpx.amr.ParallelDescriptor.MyProc() == 0:
            params.save(f"{args.outdir}/params.json")
            simulation.write_input_file(file_name=f"{args.outdir}/warpx_used_inputs")
            simulation.initialize_inputs()
            simulation.initialize_warpx()
            simulation.step(params.total_steps)
        else:
            simulation.amr_restart = args.restart
            simulation.initialize_inputs()
            simulation.initialize_warpx()
            step_number = simulation.extension.warpx.getistep(lev=0)
            simulation.step(params.total_steps - step_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", "--outdir", help="Path to output directory")
    parser.add_argument("-cpu", "--cpu", action="store_true", help="Set backend to CPU")
    parser.add_argument("-gpu", "--gpu", action="store_true", help="Set backend to GPU")
    parser.add_argument("-restart", "--restart", help="Set restart file")
    args = parser.parse_args()
    if not args.outdir:
        raise RuntimeError("Output directory must be set")
    if args.cpu == args.gpu:
        raise RuntimeError(f"Cannot set backend to both CPU and GPU to {args.cpu}")
    my_2d_mirror = MagneticMirror2D()
    my_2d_mirror.run_sim()
