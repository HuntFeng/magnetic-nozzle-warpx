from typing import Literal
import h5py
import os
import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from params import Params
from magnetic_field import CoilBField
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = "16"


class Analysis:
    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        self.params = Params()
        self.params.load(f"{dirname}/params.json")
        self.files = os.listdir(f"{self.dirname}/diag")
        self.files.remove("paraview.pmd")
        self.files.sort(key=lambda name: int(name[8:-3]))
        self.steps = np.sort([0] + [int(file[8:-3]) for file in self.files])
        self.time = self.steps * self.params.dt

        # grid for plotting
        self.r = np.linspace(0, self.params.Lr, self.params.Nr)
        self.z = np.linspace(-self.params.Lz / 2, self.params.Lz / 2, self.params.Nz)
        self.Z, self.R = np.meshgrid(self.z, self.r)

        # field data
        self.field_data = {}
        # specs (number of cores, etcs)
        self.specs = {}
        # wall time per step
        self.time_per_step = None

        # read data files
        self.extract_field_data()
        self.extract_particle_data()
        self.extract_time_data()

    def extract_field_data(self):
        """
        Return time series data

        field_data[field_name].shape = (Nr,Nz,total_steps)
        """
        for key in ["rho_electrons", "rho_ions", "phi", "part_per_cell"]:
            if key not in self.field_data:
                self.field_data[key] = np.zeros(
                    (self.params.Nr, self.params.Nz, len(self.steps))
                )
            for frame, file in enumerate(self.files, start=1):
                f = h5py.File(f"{self.dirname}/diag/{file}", "r")
                field_array = np.array(f[f"data/{self.steps[frame]}/fields/{key}"]).T
                if len(field_array.shape) == 3:
                    self.field_data[key][:, :, frame] = field_array[:, :, 0]
                elif len(field_array.shape) == 2:
                    self.field_data[key][:, :, frame] = field_array[:, :]

        self.field_data["n_electrons"] = (
            self.field_data["rho_electrons"] / -sp.constants.e
        )
        self.field_data["n_ions"] = self.field_data["rho_ions"] / sp.constants.e
        self.field_data["normed_phi"] = self.field_data["phi"] / self.params.T_e

    def extract_particle_data(self):
        for species in ["electrons", "ions"]:
            if species not in self.field_data:
                self.field_data[f"mr_{species}"] = np.zeros(
                    (self.params.Nr, self.params.Nz, len(self.steps))
                )
                self.field_data[f"mz_{species}"] = np.zeros(
                    (self.params.Nr, self.params.Nz, len(self.steps))
                )

            for frame, file in enumerate(self.files, start=1):
                f = h5py.File(f"{self.dirname}/diag/{file}", "r")
                x = np.array(
                    f[f"data/{self.steps[frame]}/particles/{species}/position/x"]
                )
                y = np.array(
                    f[f"data/{self.steps[frame]}/particles/{species}/position/y"]
                )
                z = np.array(
                    f[f"data/{self.steps[frame]}/particles/{species}/position/z"]
                )
                r = np.sqrt(x**2 + y**2)
                mx = np.array(
                    f[f"data/{self.steps[frame]}/particles/{species}/momentum/x"]
                )
                my = np.array(
                    f[f"data/{self.steps[frame]}/particles/{species}/momentum/y"]
                )
                mz = np.array(
                    f[f"data/{self.steps[frame]}/particles/{species}/momentum/z"]
                )
                mr = np.sqrt(mx**2 + my**2)

                # logical coordinate
                i = (np.sqrt(x**2 + y**2) / self.params.dr).astype(int)
                k = ((z + self.params.Lz / 2) / self.params.dz).astype(
                    int
                )  # need to shift z to positive other k is negative

                for n in range(mr.size):
                    self.field_data[f"mr_{species}"][i[n], k[n], frame] += (
                        mr[n] / mr.size
                    )
                    self.field_data[f"mz_{species}"][i[n], k[n], frame] += (
                        mz[n] / mz.size
                    )

    def extract_time_data(self):
        """
        Extarct the time steps from a standard warpx output log

        each column corresponds to the 6 numbers in the output log
        time_date[:,0]: STEP
        time_date[:,1]: TIME
        time_date[:,2]: DT
        time_date[:,3]: Evolve time
        time_date[:,4]: This step
        time_date[:,5]: Avg. per step
        """
        output_file = f"{self.dirname}/output.log"
        regex_core = re.compile(r"MPI initialized with ([0-9]*) MPI processes")
        regex_omp = re.compile(r"OMP initialized with ([0-9]*) OMP threads")
        regex_step = re.compile(
            r"STEP [0-9]* ends.*\n.* Avg\. per step = ([0-9]*[.])?[0-9]+ s",
            re.MULTILINE,
        )
        regex_mlmg = re.compile(
            r"MLMG: Final Iter.*\n.* Bottom = ([0-9]*[.])?[0-9]+", re.MULTILINE
        )
        regex_real = re.compile(r" -?[\d.]+(?:e-?\d+)?", re.MULTILINE)

        with open(output_file) as f:
            text = f.read()
            self.specs["cores"] = int(regex_core.search(text).group(1))
            self.specs["omp_threads"] = int(regex_omp.search(text).group(1))
            step_strings = [s.group(0) for s in regex_step.finditer(text)]
            mlmg_strings = [s.group(0) for s in regex_mlmg.finditer(text)]

        time_data = np.zeros([len(step_strings), 6])
        mlmg_data = np.zeros([len(step_strings), 6])
        for i, ss in enumerate(step_strings):
            numbers = regex_real.findall(ss)
            time_data[i, :] = np.array(numbers)
        for i, ss in enumerate(mlmg_strings):
            numbers = regex_real.findall(ss)
            mlmg_data[i, :] = np.array(numbers)

        self.time_per_step = time_data[:, 4]

    def plot_time(self, estimate_steps: int = 0, fitting_start: int = 0):
        """
        Plot time vs step graph.
        If estimate_steps is given, estimate the time.
        If fitting_start is given, fitting fits graph after fitting_start
        """
        plt.figure()
        steps = np.arange(1, self.time_per_step.size + 1)
        total_times = self.time_per_step.cumsum()
        plt.plot(
            steps,
            total_times,
            "o",
            label=f"Actual time {int(total_times[-1]/3600)}(hours)",
        )
        plt.xlabel("Step")
        plt.ylabel("Total time (s)")
        plt.title(f"{self.specs['cores']} Cores")

        if estimate_steps > 0:
            p = np.polyfit(
                steps[steps > fitting_start],
                total_times[steps > fitting_start],
                2,
            )
            estimate_steps = np.linspace(1, estimate_steps)
            estimate_times = np.polyval(p, estimate_steps)
            plt.plot(
                estimate_steps,
                estimate_times,
                label=f"Estimated time {int(estimate_times[-1]/3600)}(hours)",
            )
        plt.legend()
        plt.show()

        plt.figure()
        plt.semilogy(steps, self.time_per_step)
        plt.xlabel("Step")
        plt.ylabel("Time per step (s)")
        plt.title(f"{self.specs['cores']} Cores")
        plt.show()

    def plot_density(self, frame: int, plot_type: Literal["slice", "line"] = "slice"):
        time = self.time[frame]
        n_e = self.field_data["n_electrons"][:, :, frame]
        n_i = self.field_data["n_ions"][:, :, frame]
        if plot_type == "slice":
            fig, ax = plt.subplots(1, 2, sharey=True)
            pm = ax[0].pcolormesh(self.R, self.Z, n_e, cmap="Blues")
            fig.colorbar(pm, ax=ax[0], label="$n_e$")
            ax[0].set_xlabel("$r$ (m)")
            ax[0].set_ylabel("$z$ (m)")
            pm = ax[1].pcolormesh(self.R, self.Z, n_i, cmap="Reds")
            fig.colorbar(pm, ax=ax[1], label="$n_i$")
            ax[1].set_xlabel("$r$ (m)")
            ax[1].set_ylabel("$z$ (m)")
            fig.suptitle(f"$t$={time:.2e}s")
            fig.tight_layout()
        else:
            fig = plt.figure()
            plt.plot(self.z, n_e[0, :], color="blue", label="$n_e$")
            plt.plot(self.z, n_i[0, :], color="red", label="$n_i$")
            plt.xlabel("$z$ (m)")
            plt.ylabel("$\\rho$ (m$^{-3}$)")
            plt.title(f"$t$={time:.2e}s")
            plt.legend()
            fig.show()

    def plot_potential(self, frame: int, plot_type: Literal["slice", "line"] = "slice"):
        time = self.time[frame]
        normed_phi = self.field_data["normed_phi"][:, :, frame]
        if plot_type == "slice":
            fig = plt.figure()
            plt.pcolormesh(self.R, self.Z, normed_phi, cmap="coolwarm")
            plt.colorbar(label="$\\phi/T_e$")
            plt.xlabel("$r$ (m)")
            plt.ylabel("$z$ (m)")
            plt.title(f"$t$={time:.2e}s")
            fig.show()
        else:
            fig = plt.figure()
            plt.plot(self.z, normed_phi[0, :])
            plt.xlabel("$z$ (m)")
            plt.ylabel("$\\phi/T_e$ ")
            fig.show()

    def plot_momentum(self, frame: int, plot_type: Literal["slice", "line"] = "slice"):
        time = self.time[frame]
        mz_e = self.field_data["mz_electrons"][:, :, frame]
        mz_i = self.field_data["mz_ions"][:, :, frame]
        if plot_type == "slice":
            fig, ax = plt.subplots(1, 2, sharey=True)
            pm = ax[0].pcolormesh(self.R, self.Z, mz_e, cmap="Reds")
            fig.colorbar(pm, label="$m_{ze}$")
            ax[0].set_xlabel("$r$ (m)")
            ax[0].set_ylabel("$z$ (m)")
            pm = ax[1].pcolormesh(self.R, self.Z, mz_i, cmap="Reds")
            fig.colorbar(pm, label="$m_{zi}$")
            ax[1].set_xlabel("$r$ (m)")
            ax[1].set_ylabel("$z$ (m)")
            fig.suptitle(f"$t$={time:.2e}s")
            fig.tight_layout()
            fig.show()
        else:
            fig = plt.figure()
            plt.plot(self.z, mz_e[0, :], label="$m_{ze}$")
            plt.plot(self.z, mz_i[0, :], label="$m_{zi}$")
            plt.xlabel("$z$ (m)")
            plt.title(f"$t$={time:.2e}s")
            plt.legend()
            fig.show()

    def plot_magnetic_field(self):
        coil = CoilBField(R=self.params.R_coil, B_max=self.params.B_max)
        Br, Bz = coil.get_B_field(self.R, self.Z)

        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 7))
        pm = ax[0].pcolormesh(self.R, self.Z, Bz, cmap="Greens")
        fig.colorbar(pm)
        ax[0].quiver(self.R, self.Z, Br, Bz)
        ax[0].set_xlabel(f"$r$ (m)")
        ax[0].set_ylabel(f"$z$ (m)")
        ax[0].set_title("$B(r, z)$ (T)")
        ax[1].plot(Bz[0, :], self.z)
        ax[1].set_xlabel("|B| (T)")
        ax[1].set_title("B(0, z)")

        plt.tight_layout()
        fig.show()

    def animate_slice(self, field_name: Literal["density", "momentum", "potential"]):
        fig, ax = plt.subplots(1, 1 if field_name == "potential" else 2)
        if field_name == "density":
            n_e = self.field_data["n_electrons"][:, :, 0]
            n_i = self.field_data["n_ions"][:, :, 0]
            pm_e = ax[0].pcolormesh(self.R, self.Z, n_e, cmap="Blues")
            pm_i = ax[1].pcolormesh(self.R, self.Z, n_i, cmap="Reds")
            fig.colorbar(pm_e, label="$n_e$")
            fig.colorbar(pm_i, label="$n_i$")
        elif field_name == "momentum":
            mz_e = self.field_data["mz_electrons"][:, :, 0]
            mz_i = self.field_data["mz_ions"][:, :, 0]
            pm_e = ax[0].pcolormesh(self.R, self.Z, mz_e, cmap="Reds")
            pm_i = ax[1].pcolormesh(self.R, self.Z, mz_i, cmap="Reds")
            fig.colorbar(pm_e, label="$m_{ze}$")
            fig.colorbar(pm_i, label="$m_{zi}$")
        else:
            phi = self.field_data["phi"][:, :, 0]
            pm = ax.pcolormesh(self.R, self.Z, phi)
            fig.colorbar(pm, label="$phi_e$")
        time = self.time[0]
        fig.suptitle(f"$t$={time:.2e}s")
        fig.tight_layout()

        def animate(frame: int):
            if field_name == "density":
                n_e = self.field_data["n_electrons"][:, :, frame]
                n_i = self.field_data["n_ions"][:, :, frame]
                pm_e.set_array(n_e)
                pm_i.set_array(n_i)
            elif field_name == "momentum":
                mz_e = self.field_data["mz_electrons"][:, :, frame]
                mz_i = self.field_data["mz_ions"][:, :, frame]
                pm_e.set_array(mz_e)
                pm_i.set_array(mz_i)
            else:
                phi = self.field_data["phi"][:, :, frame]
                pm.set_array(phi)
            time = self.time[frame]
            fig.suptitle(f"$t$={time:.2e}s")

        anime = FuncAnimation(fig, animate, frames=tqdm(range(len(self.steps))))
        anime.save(f"{self.dirname}/slice_plot_{field_name}.mp4")

    def animate_line(self, field_name: Literal["density", "momentum", "potential"]):
        fig, ax = plt.subplots(1, 1)
        if field_name == "density":
            n_e = self.field_data["n_electrons"][0, :, 0]
            n_i = self.field_data["n_ions"][0, :, 0]
            (ln1,) = ax.plot(self.z, n_e, color="blue", label="$n_e$")
            (ln2,) = ax.plot(self.z, n_i, color="red", label="$n_i$")
        elif field_name == "momentum":
            mz_e = self.field_data["mz_electrons"][0, :, 0]
            mz_i = self.field_data["mz_ions"][0, :, 0]
            (ln1,) = ax.plot(self.z, mz_e, label="$m_{ze}$")
            (ln2,) = ax.plot(self.z, mz_i, label="$m_{zi}$")
        else:
            phi = self.field_data["phi"][0, :, 0]
            (ln,) = ax.plot(self.z, phi, label="$\\phi$")
        time = self.time[0]
        fig.suptitle(f"$t$={time:.2e}s")
        ax.legend()

        def animate(frame: int):
            if field_name == "density":
                n_e = self.field_data["n_electrons"][0, :, frame]
                n_i = self.field_data["n_ions"][0, :, frame]
                ln1.set_data(self.z, n_e)
                ln2.set_data(self.z, n_i)
            elif field_name == "momentum":
                mz_e = self.field_data["mz_electrons"][0, :, frame]
                mz_i = self.field_data["mz_ions"][0, :, frame]
                ln1.set_data(self.z, mz_e)
                ln2.set_data(self.z, mz_i)
            else:
                phi = self.field_data["phi"][0, :, frame]
                ln.set_data(self.z, phi)
            time = self.time[frame]
            fig.suptitle(f"$t$={time:.2e}s")

        anime = FuncAnimation(fig, animate, frames=tqdm(range(len(self.steps))))
        anime.save(f"{self.dirname}/line_plot_{field_name}.mp4")


if __name__ == "__main__":
    analysis = Analysis("diags202312141955")
    for field in ["density", "momentum", "potential"]:
        analysis.animate_slice(field)
        analysis.animate_line(field)
