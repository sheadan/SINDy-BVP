"""DependentVariable class provides template and utility functions.

Template and util. functions for generating candidate nonlinear terms
used in SINDy framework.
"""

# Third Party Imports
import numpy as np

# Package Imports
from .variable import Variable


class DependentVariable(Variable):
    """Define DependentVariable class and utility functions."""

    def __init__(self, name: str, data: np.ndarray, nonlinear_deg: int = 3):
        """Initialize attributes for dependent variables.

        Keyword arguments:
        name -- string used to generate descriptors for candidate fn library
        data -- numpy array with relevant data for the dependent variable
        nonlinear_deg -- maximum nonlinearity term computed
        (e.g. a nonlinear_deg of 3 means terms_dict will contain u, u^2, u^3)
        """
        super().__init__(name, data)
        self.nonlinear_deg = nonlinear_deg

    def compute_terms(self):
        """Collect and return terms for function library in a dictionary.

        Returns
        terms_dict -- dict with descriptor term keys and numpy data arrays.
        """
        terms_dict = {self.name: self.data}
        nl_terms = self.generate_nonlinear_terms()  # Replaced w decorators...
        terms_dict.update(nl_terms)
        return terms_dict

    def generate_nonlinear_terms(self):
        """Generate and return nonlinear terms for function library.

        Returns:
        nonlinear_terms -- dict containing nonlinear terms for fn lib
        """
        nonlinear_terms = {}
        for i in range(self.nonlinear_deg):
            # Prepare a description of the nonlinear term
            description = self.format_term_descriptor(i+1)
            # Aadd it as dictionary entry in the Library poly_terms dictionary
            nonlinear_terms[description] = self.data**(i+1)
        return nonlinear_terms
