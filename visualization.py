import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from params import Params
import os
from tqdm import tqdm


def phi_diag(params: Params, file_path: str):
    data = np.load(file_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 5))
    # smooth out r = 0 data by taking average from r=0 to r=1/10
    dr = params.dr
    r = np.arange(0, 5 * dr, dr)
    r_along_z = np.repeat(r, params.Nz + 1).reshape(5, -1)
    ax1.plot(
        np.linspace(-0.5, 0.5, params.Nz + 1),
        np.sum(2 * r_along_z * dr * data[:5, :] / (4 * dr), axis=0),
        label="r=0",
    )
    # ax1.plot(
    #     np.linspace(-params.Lz/2.0/params.d_e, params.Lz/2.0/params.d_e, params.Nz+1),
    #     data[params.Nr//8], label=f'x={(-7.0/16.0)*params.Lr/params.d_e:.2f}$d_e$'
    # )
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel("z/$L_z$")
    ax1.set_ylabel("e$\phi/T_e$")

    # define your scale, with white at zero
    vmin = np.min(data)
    vmax = np.max(data)
    if vmin >= 0 or vmax <= 0:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    m = ax2.imshow(
        data.T,
        cmap="coolwarm",
        aspect="equal",
        origin="lower",
        norm=norm,
        extent=[0, 1, -0.5, 0.5],
    )
    ax2.set_xlabel("r/$L_r$")
    ax2.set_ylabel("z/$L_z$")
    bar = plt.colorbar(m, ax=ax2)
    bar.set_label("e$\phi/T_e$")

    plt.tight_layout()
    plt.savefig(os.path.splitext(file_path)[0] + ".png")
    plt.close()


def rho_diag(params: Params, file_path: str):
    data = np.load(file_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 5))
    # smooth out r = 0 data by taking average from r=0 to r=1/10
    dr = params.dr
    r = np.arange(0, 5 * dr, dr)
    r_along_z = np.repeat(r, params.Nz + 1).reshape(5, -1)
    ax1.plot(
        np.linspace(-0.5, 0.5, params.Nz + 1),
        # np.mean(data[params.Nr // 2 - 5 : params.Nr // 2 + 5], axis=0),
        np.sum(2 * r_along_z * dr * data[:5, :] / (4 * dr), axis=0),
        label="r=0",
    )
    ax1.plot(
        np.linspace(-0.5, 0.5, params.Nz + 1),
        data[params.Nr // 8],
        label=f"r={(1 / 8):.2f}$L_r$",
    )
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel("z/$L_z$")
    ax1.set_ylabel("$\\rho/\\rho_0$")

    m = ax2.imshow(
        data.T,
        cmap="coolwarm",
        aspect="equal",
        origin="lower",
        extent=[0, 1, -0.5, 0.5],
    )
    ax2.set_xlabel("r/$L_r$")
    ax2.set_ylabel("z/$L_z$")
    bar = plt.colorbar(m, ax=ax2)
    bar.set_label("$\\rho/\\rho_0$")

    plt.tight_layout()
    plt.savefig(os.path.splitext(file_path)[0] + ".png")
    plt.close()


def visualize():
    params = Params()
    params.load()
    for variable in ["rho", "phi"]:
        print(f"visualizing variable: {variable}")
        for file in tqdm(os.listdir(f"diags/{variable}")):
            if os.path.splitext(file)[1] != ".npy":
                continue
            file_path = f"diags/{variable}/{file}"
            if variable == "rho":
                rho_diag(params, file_path)
            elif variable == "phi":
                phi_diag(params, file_path)


if __name__ == "__main__":
    visualize()
