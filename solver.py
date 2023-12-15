from pywarpx import picmi, callbacks, fields


class NullElectroStaticSolver(picmi.ElectrostaticSolver):
    """
    Electric field will be 0 if this solver is used.

    To use this solver simply replace the statement
    self.solver = picmi.ElectrostaticSolver(
        grid=self.grid,
        method="Multigrid",
        required_precision=1e-6,
        warpx_self_fields_verbosity=0,
    )
    to
    self.solver = picmi.ElectrostaticSolver(grid=self.grid)
    in run_simulation.py
    """

    def __init__(
        self, grid, method=None, required_precision=None, maximum_iterations=None, **kw
    ):
        super().__init__(
            grid,
            method if method is not None else "Multigrid",
            required_precision,
            maximum_iterations,
            **kw
        )
        # install the custom poisson solver (replacing the computePhi function in warpx)
        self.phi_wrapper = None
        callbacks.installpoissonsolver(self.solve)

    def initialize_inputs(self):
        super().initialize_inputs()

    def solve(self):
        """The potential is set to 0"""
        if self.phi_wrapper is None:
            self.phi_wrapper = fields.PhiFPWrapper(0, True)
        self.phi_wrapper[Ellipsis][Ellipsis] = 0.0
