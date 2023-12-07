import numpy as np
from tqdm import tqdm
import yt
import util
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from params import Params
import sys

plt.rcParams[
    "animation.ffmpeg_path"
] = "/scinet/niagara/software/2019b/opt/base/ffmpeg/3.4.2/bin/ffmpeg"

if len(sys.argv) < 2:
    ValueError("Please provide the diagnostics folder")
elif not sys.argv[1].startswith("diags"):
    ValueError("Diagnostics folder must start with diags")
else:
    # dirname = "diags202311292146"
    dirname = sys.argv[1]
specs, time_data = util.extract_data(f"{dirname}/output.log")
params = Params()
params.load(f"{dirname}/params.json")


@yt.derived_field(name=("boxlib", "n_electrons"), units="1", sampling_type="cell")
def n_electrons(field, data):
    return -data["boxlib", "rho_electrons"] / util.constants.e


@yt.derived_field(name=("boxlib", "n_ions"), units="1", sampling_type="cell")
def n_ions(field, data):
    return data["boxlib", "rho_ions"] / util.constants.e


@yt.derived_field(name=("boxlib", "normed_phi"), units="1", sampling_type="cell")
def n_ions(field, data):
    return data["boxlib", "phi"] / params.T_e


def slice_plot_anime():
    for field in fields:
        ts = yt.load(f"{dirname}/diag*")
        # slice plot
        plot = yt.SlicePlot(ts[0], "z", field, origin="native")
        plot.set_cmap(field, "coolwarm")
        plot.set_xlabel("r (cm)")
        plot.set_ylabel("z (cm)")
        fig = plot.plots[field].figure

        # animate must accept an integer frame number. We use the frame number
        # to identify which dataset in the time series we want to load
        def animate(i):
            ds = ts[i]
            plot._switch_ds(ds)
            plot.annotate_title(f"Time={time_data[i-1][1] * 1e9 : .3f}ns")

        animation = FuncAnimation(fig, animate, frames=len(ts))
        animation.save(f"{dirname}/anime_slice_plot_{field[1]}.mp4")


def line_plot_anime(number_density: bool):
    fig, ax = plt.subplots()
    z_grid = np.linspace(-params.Lz / 2, params.Lz / 2, params.Nz)
    data = np.zeros_like(z_grid)
    if number_density:
        (ln1,) = ax.semilogy(z_grid, data, label="$n_e$")
        (ln2,) = ax.semilogy(z_grid, data, label="$n_i$")
        ax.set_ylim(1e14, 1e20)
        ax.legend()
    else:
        (ln,) = ax.plot(z_grid, data)
        ax.set_ylim(-20, 20)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("$n$" if number_density else "$\\phi$")
    ax.set_title(f"Time={time_data[0][1] * 1e9 : .3f}ns")

    ts = yt.load(f"{dirname}/diag*")

    def animate(i):
        ds = ts[i]
        cg = ds.covering_grid(
            level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
        )
        ax.set_title(f"Time={time_data[i-1][1] * 1e9 : .3f}ns")
        if number_density:
            n_e = cg[fields[0]].to_ndarray()
            n_i = cg[fields[1]].to_ndarray()
            ln1.set_data(z_grid, n_e[0, :, 0])
            ln2.set_data(z_grid, n_i[0, :, 0])
            return ln1, ln2
        else:
            normed_phi = cg[fields[-1]].to_ndarray()
            ln.set_data(z_grid, normed_phi[0, :, 0])
            return ln

    animation = FuncAnimation(fig, animate, frames=tqdm(range(len(ts))))
    animation.save(
        f"{dirname}/anime_line_plot_{'n' if number_density else 'normed_phi'}.mp4"
    )


if __name__ == "__main__":
    fields = [("boxlib", "n_electrons"), ("boxlib", "n_ions"), ("boxlib", "normed_phi")]

    slice_plot_anime()
    line_plot_anime(number_density=False)
    line_plot_anime(number_density=True)
