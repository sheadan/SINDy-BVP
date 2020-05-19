"""Variable is a meta superclass for Variable objects."""

# Python imports
from abc import ABC, abstractmethod

# Third Party Import
import numpy as np


class Variable(ABC):
    """Superclass for Variable objects."""

    def __init__(self, name: str, data: np.ndarray):
        """Compute some basic attributes for the variable.

        Keyword Arguments:
        name -- string descriptor for the name of the variable
        (note: the name is used for building library term descriptors)
        data -- NumPy array of data for the variable
        """
        self.name = name
        self.data = data
        self.ndim = data.ndim
        self.shape = data.shape

    @abstractmethod
    def compute_terms(self):
        pass

    def format_term_descriptor(self, term_exponent: int):
        """Format a descriptor for polynomial and nonlinear terms.

        Keyword arguments:
        term_exponent -- the power of the polynomial or nonlinear term.
        """
        # Construct term descriptor
        description = self.name + "^{" + str(term_exponent) + "}"

        # If the exponent is one, just use the variable name
        if term_exponent == 1:
            description = self.name

        # Return the result
        return description
