"""This is the base class for Differentiator methods.

Base class defines an __init__ and a check_data method. The __init__ method
provides a basic implementation which captures the maximum differential order
to compute. The check_data method enforces that data is 1D. In the future the
latter may be relaxed.
"""

# Python Imports
from abc import ABC, abstractmethod

# Package Imports
from sindy_bvp.variables import IndependentVariable as IndVar
from sindy_bvp.variables import DependentVariable as DepVar


class BaseDifferentiator(ABC):
    """BaseDifferentiator is a base class for differentiator objects."""

    def __init__(self, diff_order: int = 3):
        """Store diff_order attribute. Default __init__ method.

        Keyword arguments:
        diff_order -- integer order of differentiation (e.g. 2 is compute u_xx)
        """
        self.diff_order = diff_order

    @abstractmethod
    def differentiate(self, ivar: IndVar, dvar: DepVar):
        """Enforce existence of differentiate method."""
        pass

    def check_data(self, dvar: DepVar):
        """Check data dimensionality by inspecting dependent variable(s) data.

        Keyword arguments:
        dv -- DependentVariable to check the data dimensionality of

        Returns:
        boolean -- boolean indicates if data is 1D (True) or not (False)
        """
        if dvar.ndim == 1:
            return True

        print("Only one-dimensional data is currently supported.")
        return False

    def format_diff_descriptor(self, iv_name: str, dv_name: str, order: int):
        """Provide formatted string descriptor for differentiated terms."""
        # Formats the differential descriptor
        if order == 1:
            descriptor = "d" + dv_name + "/d" + iv_name
        else:
            descriptor = "d^{" + str(order) + "}" + dv_name
            descriptor += "/d" + iv_name + "^{" + str(order) + "}"
        return descriptor
