import yt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams[
    "animation.ffmpeg_path"
] = "/scinet/niagara/software/2019b/opt/base/ffmpeg/3.4.2/bin/ffmpeg"
dirname = "diags128x256-crossing_time=1"
fields = [("boxlib", "rho"), ("boxlib", "phi")]
for field in fields:
    ts = yt.load(f"{dirname}/diag*")
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

    animation = FuncAnimation(fig, animate, frames=len(ts))
    animation.save(f"animation_{field[1]}.mp4")
