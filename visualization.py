import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from params import Params
import os
    
def phi_diag(params: Params, file_path: str):
    data = np.load(file_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 5))
    ax1.plot(
        np.linspace(-params.Lz/2.0/params.d_e, params.Lz /
                    2.0/params.d_e, params.Nz+1),
        np.mean(data[params.Nx//2-5:params.Nx//2+5], axis=0), label='x=0'
    )
    # ax1.plot(
    #     np.linspace(-params.Lz/2.0/params.d_e, params.Lz/2.0/params.d_e, params.Nz+1),
    #     data[params.Nx//8], label=f'x={(-7.0/16.0)*params.Lx/params.d_e:.2f}$d_e$'
    # )
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('z/$d_e$')
    ax1.set_ylabel("e$\phi/T_e$")

    # define your scale, with white at zero
    vmin = np.min(data)
    vmax = np.max(data)
    if vmin >= 0 or vmax <= 0:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    m = ax2.imshow(
        data.T, cmap='RdBu', aspect='equal', origin='lower',
        norm=norm,
        extent=[
            -params.Lx/2.0/params.d_e, params.Lx/2.0/params.d_e,
            -params.Lz/2.0/params.d_e, params.Lz/2.0/params.d_e
        ]
    )
    ax2.set_xlabel('x/$d_e$')
    ax2.set_ylabel("z/$d_e$")
    bar = plt.colorbar(m, ax=ax2)
    bar.set_label("e$\phi/T_e$")

    plt.tight_layout()
    plt.savefig(os.path.splitext(file_path)[0]+".png")
    plt.close()

def rho_diag(params: Params, file_path: str):
    data = np.load(file_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 5))
    ax1.plot(
        np.linspace(-params.Lz/2.0/params.d_e, params.Lz /
                    2.0/params.d_e, params.Nz+1),
        np.mean(data[params.Nx//2-5:params.Nx//2+5], axis=0), label='x=0'
    )
    ax1.plot(
        np.linspace(-params.Lz/2.0/params.d_e, params.Lz /
                    2.0/params.d_e, params.Nz+1),
        data[params.Nx //
             8], label=f'x={(-7.0/16.0)*params.Lx/params.d_e:.2f}$d_e$'
    )
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('z/$d_e$')
    ax1.set_ylabel("$n_i/n_0$")

    m = ax2.imshow(
        data.T, cmap='Greens', aspect='equal', origin='lower',
        extent=[
            -params.Lx/2.0/params.d_e, params.Lx/2.0/params.d_e,
            -params.Lz/2.0/params.d_e, params.Lz/2.0/params.d_e
        ]
    )
    ax2.set_xlabel('x/$d_e$')
    ax2.set_ylabel("z/$d_e$")
    bar = plt.colorbar(m, ax=ax2)
    bar.set_label("$n_i/n_0$")

    plt.tight_layout()
    plt.savefig(os.path.splitext(file_path)[0]+".png")
    plt.close()

def visualize():
    params = Params()
    params.load()
    for variable in ["rho", "phi"]:
        for file in os.listdir(f"diags/{variable}"):
            file_path = f"diags/{variable}/{file}"
            if variable == "rho":
                rho_diag(params, file_path)
            elif variable == "phi":
                phi_diag(params, file_path)

if __name__ == "__main__":
    visualize()