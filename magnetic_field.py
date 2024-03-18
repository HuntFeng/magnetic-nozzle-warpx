from typing import Literal
import numpy as np
import scipy.constants as consts


class CoilBField(object):
    """Class to calculate the magnetic field strength due to a current carrying
    wire loop using Eqs. 64 and 65 from https://arxiv.org/pdf/2002.09444.pdf.

    Arguments:
        R: (float): Radius of the wire loop, in meter.
        I (float): Direct current through the loop in A, can only be specified
            if ratio is not given.
        B_max (float): Minimum value of magnetic field strength in z (assumed
            background field), in Tesla.
    """

    def __init__(self, R, I=None, B_max=None, verbose=False):
        self.R = R
        self.I = I
        self.B_max = B_max

        if self.I is not None and self.B_max is not None:
            raise AttributeError("Only I or B_max can be specified.")

        # if I is specified calculate the mirror ratio, otherwise calculate I
        if self.I is not None:
            self.B_max = self.get_B_field(0.0, 0.0)[1]
        else:
            self.I = 2.0 * self.R * self.B_max / consts.mu_0

        if verbose:
            print(f"Coil radius is {self.R} m")
            print(f"Coil current is {self.I*1e-3:.1f} kA")
            print(f"Maximum B-field is {self.B_max:.1f} T")

    def get_B_field(self, r, z):
        """Function to return the B field components at coordinate (r, z)."""
        B_r = (
            consts.mu_0
            * self.I
            / 4.0
            * r
            * (3.0 * self.R**2 * z / (self.R**2 + z**2) ** (5.0 / 2))
        )
        B_z = (
            consts.mu_0
            * self.I
            / 2.0
            * (
                self.R**2 / (self.R**2 + z**2) ** (3.0 / 2)
                - 3.0 / 2.0 * self.R**2 * r**2 / (self.R**2 + z**2) ** (5.0 / 2)
            )
        )
        return B_r, B_z

    def get_Bx_expression(self):
        """Just a utility function that can be queried to get the magnetic
        field strength in the x-direction, as input for WarpX."""
        return (
            f"{consts.mu_0 * self.I / 4.0}*x*("
            f"{3.0 * self.R**2}*z/({self.R**2}+z**2)**2.5"
            ")"
        )

    def get_By_expression(self):
        """Just a utility function that can be queried to get the magnetic
        field strength in the y-direction, as input for WarpX."""
        return (
            f"{consts.mu_0 * self.I / 4.0}*y*("
            f"{3.0 * self.R**2}*z/({self.R**2}+z**2)**2.5"
            ")"
        )

    def get_Bz_expression(self):
        """Just a utility function that can be queried to get the magnetic
        field strength in the z-direction, as input for WarpX."""
        return (
            f"{consts.mu_0 * self.I / 2.0}*("
            f"{self.R**2}/({self.R**2}+z**2)**1.5"
            f"-{3.0 / 2.0 * self.R**2}*(x**2 + y**2)/({self.R**2}+z**2)**2.5"
            ")"
        )


class NozzleBField:
    """Class to calculate the magnetic field strength

    Arguments:
        B0: Maximum magnetic field at the nozzle throat in Tesla
        R: Mirror ratio for the converging part
        K: Expansion ratio for the diverging part
        rappa: Correction factor for converging part
        kappa: Correction factor for diverging part
    """

    def __init__(
        self,
        B_max: float,
        R: float,
        K: float,
        kappa: float,
        rappa: float,
        verbose=False,
    ):
        self.B_max = B_max
        self.R = R
        self.K = K
        self.kappa = kappa
        self.rappa = rappa
        self.z0 = 0.5

        if verbose:
            print(f"Maximum B-field is {self.B_max} T")
            print(f"Mirror ratio {self.R}")
            print(f"Expansion ratio {self.K}")
            print(f"Correction factor for converging part is {self.rappa}")
            print(f"Correction factor for diverging part is {self.kappa}")

    def get_B_field(self, r, z):
        """Function to return the B field components at coordinate (r, z)."""
        B_max, R, K, kappa, rappa, z0 = (
            self.B_max,
            self.R,
            self.K,
            self.kappa,
            self.rappa,
            self.z0,
        )

        B_r = B_max * r * (rappa * (R - 1) * z0**2 * z) / (
            (R * rappa - 1) * z**2 + z0**2
        ) ** 2 * np.heaviside(-z, 0.5) + r * (kappa * (K - 1) * z0**2 * z) / (
            (K * kappa - 1) * z**2 + z0**2
        ) ** 2 * np.heaviside(
            z, 0.5
        )
        B_z = B_max * (1.0 + (rappa - 1) * (z / z0) ** 2) / (
            1 + (R * rappa - 1) * (z / z0) ** 2
        ) * np.heaviside(-z, 0.5) + (1.0 + (kappa - 1) * (z / z0) ** 2) / (
            1 + (K * kappa - 1) * (z / z0) ** 2
        ) * np.heaviside(
            z, 0.5
        )
        return B_r, B_z

    def get_Bx_expression(self, region=Literal["mirror", "expander"]):
        """Just a utility function that can be queried to get the magnetic
        field strength in the x-direction, as input for WarpX."""
        B_max, R, K, kappa, rappa, z0 = (
            self.B_max,
            self.R,
            self.K,
            self.kappa,
            self.rappa,
            self.z0,
        )
        match region:
            case "mirror":
                return (
                    f"{B_max} * x * ({rappa * (R - 1) * z0**2} * z) / ("
                    f"{(R * rappa - 1)} * z**2 + {z0**2}"
                    ") ** 2"
                )
            case "expander":
                return (
                    f"{B_max} * x * ({kappa * (K - 1) * z0**2} * z) / ("
                    f"{(K * kappa - 1)} * z**2 + {z0**2}"
                    ") ** 2"
                )
            case _:
                raise TypeError()

    def get_By_expression(self, region=Literal["mirror", "expander"]):
        """Just a utility function that can be queried to get the magnetic
        field strength in the y-direction, as input for WarpX."""
        B_max, R, K, kappa, rappa, z0 = (
            self.B_max,
            self.R,
            self.K,
            self.kappa,
            self.rappa,
            self.z0,
        )
        match region:
            case "mirror":
                return (
                    f"{B_max} * y * ({rappa * (R - 1) * z0**2} * z) / ("
                    f"{(R * rappa - 1)} * z**2 + {z0**2}"
                    ") ** 2"
                )
            case "expander":
                return (
                    f"{B_max} * y * ({kappa * (K - 1) * z0**2} * z) / ("
                    f"{(K * kappa - 1)} * z**2 + {z0**2}"
                    ") ** 2"
                )
            case _:
                raise TypeError()

    def get_Bz_expression(self, region=Literal["mirror", "expander"]):
        """Just a utility function that can be queried to get the magnetic
        field strength in the z-direction, as input for WarpX."""
        B_max, R, K, kappa, rappa, z0 = (
            self.B_max,
            self.R,
            self.K,
            self.kappa,
            self.rappa,
            self.z0,
        )
        match region:
            case "mirror":
                return (
                    f"{B_max} * (1.0 + ({rappa} - 1) * (z / {z0}) ** 2) / ("
                    f"1 + ({R * rappa} - 1) * (z / {z0}) ** 2"
                    ")"
                )
            case "expander":
                return (
                    f"{B_max} * (1.0 + ({kappa} - 1) * (z / {z0}) ** 2) / ("
                    f"1 + ({K * kappa} - 1) * (z / {z0}) ** 2"
                    ")"
                )
            case _:
                raise TypeError()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    B_max = 1
    R = 10
    K = 50
    kappa = 1.0
    rappa = 5.0
    nozzle = NozzleBField(B_max, R, K, kappa, rappa, True)
    r_grid = np.linspace(0, 0.1, 250)
    z_grid = np.linspace(-0.5, 0.5, 500)
    rr, zz = np.meshgrid(r_grid, z_grid)
    Br, Bz = nozzle.get_B_field(rr, zz)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(5, 7))
    pm = ax.pcolormesh(rr, zz, Bz, cmap="Greens")  # ,norm=colors.LogNorm())
    fig.colorbar(pm)
    ax.streamplot(rr, zz, Br, Bz, density=2, color="black", minlength=1, linewidth=0.75)
    ax.set_xlabel(f"$r$ (m)")
    ax.set_ylabel(f"$z$ (m)")
    ax.set_title("$B(r, z)$ (T)")
    ax.set_xlim(r_grid.min(), r_grid.max())
    ax.set_ylim(z_grid.min(), z_grid.max())
    plt.savefig("magnetic_field.png")
