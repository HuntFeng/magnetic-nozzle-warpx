import h5py
import sys
import os
import re
import glob
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Literal
from params import Params
from magnetic_field import CoilBField
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = "16"

Species = Literal["electrons", "ions"]
FieldKey = Literal[
    "n_electrons",
    "n_ions",
    "rho_electrons",
    "rho_ions",
    "mz_electrons",
    "mz_ions",
    "mr_electrons",
    "mr_ions",
    "phi",
    "normed_phi",
    "part_per_cell",
]
Direction = Literal["r", "z"]
PlotType = Literal["slice", "line"]


class Analysis:
    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        self.params = Params()
        self.params.load(f"{dirname}/params.json")
        self.files = os.listdir(f"{self.dirname}/diag")
        if "paraview.pmd" in self.files:
            self.files.remove("paraview.pmd")
        self.files.sort(key=lambda name: int(name[8:-3]))
        self.steps = np.sort([int(file[8:-3]) for file in self.files])
        self.time = self.steps * self.params.dt

        # grid for plotting
        self.r = np.linspace(0, self.params.Lr, self.params.Nr)
        self.z = np.linspace(-self.params.Lz / 2, self.params.Lz / 2, self.params.Nz)
        self.Z, self.R = np.meshgrid(self.z, self.r)

    # TODO: too slow, improve this
    def extract_momentum(
        self, f: h5py.File, species: Species, frame: int, direction: Literal["r", "z"]
    ):
        """Get momentum of species at certain frame"""
        x = np.array(f[f"data/{self.steps[frame]}/particles/{species}/position/x"])
        y = np.array(f[f"data/{self.steps[frame]}/particles/{species}/position/y"])
        z = np.array(f[f"data/{self.steps[frame]}/particles/{species}/position/z"])
        r = np.sqrt(x**2 + y**2)
        mx = np.array(f[f"data/{self.steps[frame]}/particles/{species}/momentum/x"])
        my = np.array(f[f"data/{self.steps[frame]}/particles/{species}/momentum/y"])
        mz = np.array(f[f"data/{self.steps[frame]}/particles/{species}/momentum/z"])
        mr = np.sqrt(mx**2 + my**2)
        w = np.array(f[f"data/{self.steps[frame]}/particles/{species}/weighting"])

        # logical coordinate
        i = (r / self.params.dr).astype(int)
        # need to shift z to positive other k is negative
        k = ((z + self.params.Lz / 2) / self.params.dz).astype(int)

        data = np.zeros((self.params.Nr, self.params.Nz))
        for n in range(mr.size):
            if direction == "r":
                data[i[n], k[n]] += w[n] * mr[n] / mr.size
            else:
                data[i[n], k[n]] += w[n] * mz[n] / mz.size
        return data

    def get_data(self, key: FieldKey, frame: int):
        """Get data with field key at certain frame"""
        file = h5py.File(f"{self.dirname}/diag/{self.files[frame]}", "r")
        match key:
            case "rho_electrons":
                data = np.array(file[f"data/{self.steps[frame]}/fields/{key}"])
                return data.T[:, :, 0]
            case "rho_ions":
                data = np.array(file[f"data/{self.steps[frame]}/fields/{key}"])
                return data.T[:, :, 0]
            case "n_electrons":
                data = np.array(file[f"data/{self.steps[frame]}/fields/rho_electrons"])
                return data.T[:, :, 0] / -sp.constants.e
            case "n_ions":
                data = np.array(file[f"data/{self.steps[frame]}/fields/rho_ions"])
                return data.T[:, :, 0] / sp.constants.e
            case "phi":
                data = np.array(file[f"data/{self.steps[frame]}/fields/phi"])
                return data.T[:, :, 0]
            case "normed_phi":
                data = np.array(file[f"data/{self.steps[frame]}/fields/phi"])
                return data.T[:, :, 0] / self.params.T_e
            case "part_per_cell":
                # this field has exactly 2 dimensions
                return np.array(file[f"data/{self.steps[frame]}/fields/{key}"]).T
            case _:
                if not key.startswith("m"):
                    raise KeyError()
                direction = key[1]
                species = key.split("_")[1]
                return self.extract_momentum(file, species, frame, direction)

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

        regex_core = re.compile(r"MPI initialized with ([0-9]*) MPI processes")
        regex_omp = re.compile(r"OMP initialized with ([0-9]*) OMP threads")
        regex_device = re.compile(r"CUDA initialized with ([0-9]*) device")
        regex_step = re.compile(
            r"STEP [0-9]* ends.*\n.* Avg\. per step = ([0-9]*[.])?[0-9]+ s",
            re.MULTILINE,
        )
        regex_mlmg = re.compile(
            r"MLMG: Final Iter.*\n.* Bottom = ([0-9]*[.])?[0-9]+", re.MULTILINE
        )
        regex_real = re.compile(r" -?[\d.]+(?:e-?\d+)?", re.MULTILINE)

        specs = {}
        file_list = glob.glob(f"{self.dirname}/*.log")
        if len(file_list) == 0:
            raise FileNotFoundError(
                "The standard output from WarpX must be placed in diags folder and have extension .log"
            )
        elif len(file_list) > 1:
            raise RuntimeError("Too many output logs.")
        else:
            output_file = file_list[0]

        with open(output_file) as f:
            text = f.read()
            for key, regex in [
                ("cores", regex_core),
                ("omps", regex_omp),
                ("devices", regex_device),
            ]:
                match = regex.search(text)
                if match:
                    specs[key] = int(match.group(1))
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

        return specs, time_data

    def plot_time(self, total_step: int = None, fitting_range: range = None):
        """
        Plot time vs step graph.
        If total_step is given, estimate the time.
        If fitting_range is given, fit the graphs within the range
        """
        specs, time_data = self.extract_time_data()
        steps = time_data[:, 0]
        total_times = time_data[:, 3]
        time_per_step = time_data[:, 4]
        # remove outputing steps
        remove_output = (steps.astype(int) % int(self.params.diag_steps)) != 0
        steps = steps[remove_output]
        total_times = total_times[remove_output]
        time_per_step = time_per_step[remove_output]

        # time per step
        plt.figure()
        plt.semilogy(steps, time_per_step)
        plt.xlabel("Step")
        plt.ylabel("Time per step (s)")
        if "devices" in specs:
            plt.title(f"{specs['devices']} Devices")
        elif "cores" in specs:
            plt.title(f"{specs['cores']} Cores")
        if total_step is not None or fitting_range is not None:
            time_per_step_p = np.polyfit(
                steps[fitting_range or Ellipsis],
                np.log(time_per_step[fitting_range or Ellipsis]),
                1,
            )
            plt.semilogy(
                steps,
                np.exp(np.polyval(time_per_step_p, steps)),
                "--",
                label=f"slope={time_per_step_p[0]:.1e}",
            )
            plt.legend()
        plt.show()

        # total times
        plt.figure()
        plt.plot(
            steps,
            total_times,
            "o",
            # use self.steps here because the last step was removed
            label=f"{int(self.steps[-1])} steps takes {total_times[-1]/3600:.2f}(hours)",
        )
        plt.xlabel("Step")
        plt.ylabel("Total time (s)")
        if "devices" in specs:
            plt.title(f"{specs['devices']} Devices")
        elif "cores" in specs:
            plt.title(f"{specs['cores']} Cores")
        if total_step is not None and total_step > 0 and time_per_step_p[0] < 1e-3:
            print(
                "Slope is small, consider time per step as constant. \nUsing linear fitting."
            )
            p = np.polyfit(
                steps[fitting_range or Ellipsis],
                total_times[fitting_range or Ellipsis],
                1,
            )
            estimate_steps = np.linspace(1, total_step)
            estimate_times = np.polyval(p, estimate_steps)
            plt.plot(
                estimate_steps,
                estimate_times,
                label=f"{int(estimate_steps[-1])} steps will take {estimate_times[-1]/3600:.2f}(hours)",
            )
        plt.legend()
        plt.show()

    def plot_part_per_cell(self, frame: int):
        time = self.time[frame]
        data = self.get_data("part_per_cell", frame)
        print(f"Total particles: {data.sum()}")
        plt.figure()
        plt.pcolormesh(self.R, self.Z, data, cmap="Reds", norm="symlog")
        plt.colorbar(label="Particle Per Cell")
        plt.xlabel("$r$ (m)")
        plt.ylabel("$z$ (m)")
        plt.title(f"$t$={time:.2e}s")
        plt.show()

    def average_along_central_axis(self, data: np.array):
        dr = self.params.dr
        Nz = self.params.Nz
        R = 5 * dr
        r_along_z = np.repeat(np.arange(dr / 2, R, dr), Nz).reshape(5, -1)
        # \int_0^R 2\pi rdr f(r) / \pi R^2
        return np.sum(2 * r_along_z * dr * data[:5, :] / R**2, axis=0)

    def plot_density(self, frame: int, plot_type: PlotType = "slice"):
        time = self.time[frame]
        n_e = self.get_data("n_electrons", frame)
        n_i = self.get_data("n_ions", frame)
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
            mean_n_e = self.average_along_central_axis(n_e)
            mean_n_i = self.average_along_central_axis(n_i)
            plt.semilogy(self.z, mean_n_e, color="blue", label="$n_e$")
            plt.semilogy(self.z, mean_n_i, color="red", label="$n_i$")
            plt.xlabel("$z$ (m)")
            plt.ylabel("$n$ (m$^{-3}$)")
            plt.title(f"$t$={time:.2e}s")
            plt.legend()
            fig.show()

    def plot_potential(self, frame: int, plot_type: PlotType = "slice"):
        time = self.time[frame]
        normed_phi = self.get_data("normed_phi", frame)
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
            plt.title(f"$t$={time:.2e}s")
            fig.show()

    def plot_momentum(
        self,
        frame: int,
        direction: Direction = "z",
        plot_type: PlotType = "slice",
    ):
        time = self.time[frame]
        m_e = self.get_data(f"m{direction}_electrons", frame)
        m_i = self.get_data(f"m{direction}_ions", frame)
        if plot_type == "slice":
            fig, ax = plt.subplots(1, 2, sharey=True)
            pm = ax[0].pcolormesh(self.R, self.Z, m_e, cmap="Reds")
            fig.colorbar(pm, label="$m_{ze}$")
            ax[0].set_xlabel("$r$ (m)")
            ax[0].set_ylabel("$z$ (m)")
            pm = ax[1].pcolormesh(self.R, self.Z, m_i, cmap="Reds")
            fig.colorbar(pm, label="$m_{zi}$")
            ax[1].set_xlabel("$r$ (m)")
            ax[1].set_ylabel("$z$ (m)")
            fig.suptitle(f"$t$={time:.2e}s")
            fig.tight_layout()
            fig.show()
        else:
            fig = plt.figure()
            plt.plot(self.z, m_e[0, :], label="$m_{ze}$")
            plt.plot(self.z, m_i[0, :], label="$m_{zi}$")
            plt.xlabel("$z$ (m)")
            plt.ylabel("$m$ (kg$\cdot$m/s)")
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

    def animate_slice(self, field_type: Literal["density", "momentum", "potential"]):
        fig, ax = plt.subplots(1, 1 if field_type == "potential" else 2)
        frame = 0
        if field_type == "density":
            n_e = self.get_data("n_electrons", frame)
            n_i = self.get_data("n_ions", frame)
            pm_e = ax[0].pcolormesh(self.R, self.Z, n_e, cmap="Blues")
            pm_i = ax[1].pcolormesh(self.R, self.Z, n_i, cmap="Reds")
            fig.colorbar(pm_e, label="$n_e$")
            fig.colorbar(pm_i, label="$n_i$")
        elif field_type == "momentum":
            mz_e = self.get_data("mz_electrons", frame)
            mz_i = self.get_data("mz_ions", frame)
            pm_e = ax[0].pcolormesh(self.R, self.Z, mz_e, cmap="Reds")
            pm_i = ax[1].pcolormesh(self.R, self.Z, mz_i, cmap="Reds")
            fig.colorbar(pm_e, label="$m_{ze}$")
            fig.colorbar(pm_i, label="$m_{zi}$")
        else:
            phi = self.get_data("phi", frame)
            pm = ax.pcolormesh(self.R, self.Z, phi)
            fig.colorbar(pm, label="$phi_e$")
        time = self.time[frame]
        fig.suptitle(f"$t$={time:.2e}s")
        fig.tight_layout()

        def animate(frame: int):
            if field_type == "density":
                n_e = self.get_data("n_electrons", frame)
                n_i = self.get_data("n_ions", frame)
                pm_e.set_array(n_e)
                pm_i.set_array(n_i)
            elif field_type == "momentum":
                mz_e = self.get_data("mz_electrons", frame)
                mz_i = self.get_data("mz_ions", frame)
                pm_e.set_array(mz_e)
                pm_i.set_array(mz_i)
            else:
                phi = self.get_data("phi", frame)
                pm.set_array(phi)
            time = self.time[frame]
            fig.suptitle(f"$t$={time:.2e}s")

        anime = FuncAnimation(fig, animate, frames=tqdm(range(len(self.steps))))
        anime.save(f"{self.dirname}/slice_plot_{field_type}.mp4")

    def animate_line(self, field_type: Literal["density", "momentum", "potential"]):
        fig, ax = plt.subplots(1, 1)
        frame = 0
        if field_type == "density":
            n_e = self.get_data("n_electrons", frame)[0, :]
            n_i = self.get_data("n_ions", frame)[0, :]
            (ln1,) = ax.plot(self.z, n_e, color="blue", label="$n_e$")
            (ln2,) = ax.plot(self.z, n_i, color="red", label="$n_i$")
        elif field_type == "momentum":
            mz_e = self.get_data("mz_electrons", frame)[0, :]
            mz_i = self.get_data("mz_ions", frame)[0, :]
            (ln1,) = ax.plot(self.z, mz_e, label="$m_{ze}$")
            (ln2,) = ax.plot(self.z, mz_i, label="$m_{zi}$")
        else:
            phi = self.get_data("phi", frame)[0, :]
            (ln,) = ax.plot(self.z, phi, label="$\\phi$")
        time = self.time[frame]
        fig.suptitle(f"$t$={time:.2e}s")
        ax.legend()

        def animate(frame: int):
            if field_type == "density":
                n_e = self.get_data("n_electrons", frame)[0, :]
                n_i = self.get_data("n_ions", frame)[0, :]
                ln1.set_data(self.z, n_e)
                ln2.set_data(self.z, n_i)
            elif field_type == "momentum":
                mz_e = self.get_data("mz_electrons", frame)[0, :]
                mz_i = self.get_data("mz_ions", frame)[0, :]
                ln1.set_data(self.z, mz_e)
                ln2.set_data(self.z, mz_i)
            else:
                phi = self.get_data("phi", frame)[0, :]
                ln.set_data(self.z, phi)
            time = self.time[frame]
            fig.suptitle(f"$t$={time:.2e}s")

        anime = FuncAnimation(fig, animate, frames=tqdm(range(len(self.steps))))
        anime.save(f"{self.dirname}/line_plot_{field_type}.mp4")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need 1 argument: the name of the diagnostics folder")
    else:
        dirname = sys.argv[1]
        analysis = Analysis(dirname)
        print("Making animes")
        for field in ["density", "potential"]:
            analysis.animate_slice(field)
            analysis.animate_line(field)
        print(f"Check animes in {dirname}")
