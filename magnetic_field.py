import numpy as np
import matplotlib.pyplot as plt
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
        B_r = consts.mu_0 * self.I / 4.0 * r * (
            3.0 * self.R**2 * z / (self.R**2 + z**2)**(5.0/2)
        )
        B_z = consts.mu_0 * self.I / 2.0 * (
            self.R**2 / (self.R**2 + z**2)**(3.0/2)
            - 3.0 / 2.0 * self.R**2 * r**2 / (self.R**2 + z**2)**(5.0/2)
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

    def get_Bz_expression(self):
        """Just a utility function that can be queried to get the magnetic
        field strength in the z-direction, as input for WarpX."""
        return (
            f"{consts.mu_0 * self.I / 2.0}*("
            f"{self.R**2}/({self.R**2}+z**2)**1.5"
            f"-{3.0 / 2.0 * self.R**2}*x**2/({self.R**2}+z**2)**2.5"
            ")"
        )

    def plot_field(self, Lx=1.0, Lz=2.25):
        """Function to plot the magnetic field strength."""
        x_grid = np.linspace(-Lx, Lx, 20)
        z_grid = np.linspace(-Lz, Lz, 30)

        # rr, zz = np.meshgrid(r_grid, z_grid)
        # Br, Bz = self.get_B_field(rr*self.R, zz*self.R)
        xx, zz = np.meshgrid(x_grid, z_grid)
        Bx, Bz = self.get_B_field(xx*self.R, zz*self.R)

        # creating plot
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 5.5))

        ax1.quiver(xx, zz, Bx, Bz)
        ax1.set_xlabel('x/R')
        ax1.set_ylabel('z/R')
        # ax1.set_aspect('equal')
        ax1.set_title("B(r, z)")

        c = ax1.imshow(
            Bz, cmap='Greens',
            extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
            aspect='equal', origin='lower'
        )
        #bar = plt.colorbar(c, ax=ax1, location='left')
        #bar.set_label("$B_z$ (T)")

        z_grid = np.linspace(z_grid[0], z_grid[-1], 800)
        Br, Bz = self.get_B_field(0, z_grid*self.R)
        ax2.plot(Bz, z_grid)
        ax2.set_xlabel("|B| (T)")
        ax2.set_title("B(0, z)")

        # show plot
        plt.tight_layout()
        plt.savefig('B_field.png')
        plt.show()

        # x_grid = np.linspace(-1.0, 1.0, 800)
        # Br, Bz = self.get_B_field(abs(x_grid)*self.R, 0.0)
        # plt.plot(x_grid, Bz)
        # plt.show()

# my_field = CoilBField(R=0.2, I=12.73e3)
# # my_field = CoilBField(R=0.2, ratio=5.0)
# my_field.plot_field()