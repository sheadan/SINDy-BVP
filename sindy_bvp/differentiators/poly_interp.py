"""finite_differences.py provides the FiniteDifferences Differentiator class.

The FiniteDifferences class implements finite differences method from
methods.py module.
"""
# Third Party Imports
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev as Cheb

# Package Imports
from .base_differentiator import BaseDifferentiator
from sindy_bvp.variables import IndependentVariable as IndVar
from sindy_bvp.variables import DependentVariable as DepVar


class PolyInterp(BaseDifferentiator):
    """Differentiator that implements Chebychev polynomial interpolation."""

    def __init__(self, diff_order: int = 3, width: int = 5, degree: int = 3):
        """Store diff_order and other polynomial interpolation attributes.

        Keyword arguments:
        diff_order -- max order differential to compute (e.g. 2 is u_xx)
        cheb_width -- Width of window on which to interpolate polynomial
        cheb_degree -- Polynomial degree for Chebychev polynomial fitting
        """
        super(PolyInterp, self).__init__(diff_order)
        self.cheb_width = width
        self.cheb_degree = degree

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
        differentiated_terms = self.PolyDiff(dvar.data, ivar.data)

        # Now, iterate through the differentiated terms
        for i in range(differentiated_terms.shape[1]):
            # Generate a descriptor for each term
            diff_data = differentiated_terms[:, i].flatten()
            # Becausese polynomial interpolation throws out data at the edges
            # of the data set, we pad the differential with NaNs so the data is
            # of the same length as the original data set
            diff_data = np.pad(diff_data, (self.cheb_width, self.cheb_width),
                               'constant', constant_values=np.NaN)
            # Add the differentiated data to the diff_terms dict
            descriptor = self.format_diff_descriptor(ivar.name, dvar.name, i+1)
            diff_terms[descriptor] = diff_data

        return diff_terms

    def PolyDiff(self, u, x):
        """Differentiate data with polynomial interpolation.

        This method takes a collection of points of size (2*self.cheb_width),
        fits a Chebychev polynomial to the collection of points, and computes
        the derivative at the central point. The derivatives are collated and
        returned to the calling functions as a NumPy array.

        Keyword arguments:
        u -- values of some function
        x -- corresponding x-coordinates where u is evaluated

        Note: This throws out the data close to the edges since the polynomial
        derivative only works well when we're looking at the middle of the
        points fit.
        """
        u = u.flatten()
        x = x.flatten()

        n = len(x)

        # Initialize a numpy array for storing the derivative values
        du = np.zeros((n - 2*self.cheb_width, self.diff_order))

        # Loop for fitting Cheb polynomials to each point
        for j in range(self.cheb_width, n-self.cheb_width):
            # Select the points on which to fit polynomial
            points = np.arange(j - self.cheb_width, j + self.cheb_width)
            # Fit the polynomial
            poly = Cheb.fit(x[points], u[points], self.cheb_degree)
            # Take derivatives
            for d in range(1, self.diff_order + 1):
                du[j-self.cheb_width, d-1] = poly.deriv(m=d)(x[j])

        return du
