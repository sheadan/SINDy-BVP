"""finite_differences.py provides the FiniteDifferences Differentiator class.

The FiniteDifferences class implements finite differences method from
methods.py module.
"""
# Third Party Imports
import numpy as np

# Package Imports
from .base_differentiator import BaseDifferentiator
from sindy_bvp.variables import IndependentVariable as IndVar
from sindy_bvp.variables import DependentVariable as DepVar


class FiniteDifferences(BaseDifferentiator):
    """Differentiator class which implements Finite Differences."""

    def differentiate(self, ivar: IndVar, dvar: DepVar):
        """Differentiate the dependent variable data with respect to ind var.

        Keyword arguments:
        ivar -- IndependentVariable instance to differentiate w.r.t.
        dvar -- DependentVariable instance to differentiate

        Returns:
        diff_terms -- dictionary of numerically differentiated terms
        """
        # Initialize a dictionary for the terms
        diff_terms = {}

        # Prepare differential terms with the configured differentiation method
        differentiated_terms = self.finite_differences(dvar.data,
                                                       ivar.spacing,
                                                       self.diff_order)

        # Now, iterate through the differentiated terms
        for i in range(differentiated_terms.shape[1]):
            # Generate a descriptor for each term
            descriptor = self.format_diff_descriptor(ivar.name, dvar.name, i+1)
            diff_data = differentiated_terms[:, i].flatten()
            # And add the data to the diff_terms dict
            diff_terms[descriptor] = diff_data

        return diff_terms

    def FiniteDiff(self, u, dx, d):
        """Take d-th derivative data using 2nd order finite difference method.

        Works but with poor accuracy for d > 3
        Input:
        u = data to be differentiated
        dx = Grid spacing.  Assumes uniform spacing
        """
        n = u.size
        # ux = np.zeros((n,1), dtype=np.complex64)
        ux = np.zeros((n, 1), dtype=np.float64)

        if d == 1:
            for i in range(1, n-1):
                ux[i] = (u[i+1]-u[i-1]) / (2*dx)

            ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
            ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
            return ux

        if d == 2:
            for i in range(1, n-1):
                ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2

            ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
            ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
            return ux

        if d == 3:
            for i in range(2, n-2):
                ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3

            ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
            ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
            ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5])/dx**3
            ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6])/dx**3
            return ux

        if d > 3:
            return self.FiniteDiff(self.FiniteDiff(u, dx, 3), dx, d-3)

    def finite_differences(self, u, dx, d):
        """Compute derivatives up to order d iteratively.

        Keyword arguments:
        u -- the signal, an np.array, to compute derivatives of
        dx -- the spacing of the independent variable between entries in u
        d -- the maximum order derivative to compute

        Returns:
        derivs -- an np.hstack of size (d,n) where n is length of signal u
        and each row is increasing derivatives (e.g. [u_x, u_xx, ...])
        """
        derivs = []
        for i in range(d):
            dudx = self.FiniteDiff(u, dx, i+1)
            derivs.append(dudx)
        derivs = np.hstack(derivs)
        return derivs
