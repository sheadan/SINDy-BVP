"""IndependentVariable class provides template and utility functions.

Template and util. functions for generating candidate polynomial terms
used in SINDy framework.
"""

# Third Party Import
import numpy as np

# Package Imports
from .variable import Variable


class IndependentVariable(Variable):
    """Define IndependentVariable class and utility functions."""

    def __init__(self, name: str, data: np.ndarray, poly_deg: int = 3):
        """Initialize attributes for independent variables.

        Keyword arguments:
        name -- string used to generate descriptors for candidate fn library
        data -- numpy array with relevant data for the dependent variable
        poly_deg -- maximum order polynomial term computed
        (e.g. a poly_deg of 3 means terms_dict will contain x, x^2, x^3)
        """
        super().__init__(name, data)
        self.poly_deg = poly_deg

        # Calculate the spacing between discrete points of the independent
        # variable (e.g. dx for variable 'x')
        self.spacing = np.diff(data)[0]
        # Check that spacing is the same everywhere - haven't implemented
        # uneven spacing for numerical differentiation.
        if np.allclose(self.spacing, np.diff(data)) is not True:
            print("Error in computing spacing - points are not evenly spaced!")

    def compute_terms(self):
        """Collect and return terms for function library in a dictionary.

        Returns
        terms_dict -- dict with descriptor term keys and numpy data arrays.
        """
        terms_dict = {self.name: self.data}
        poly_terms = self.generate_poly_terms()  # Replaced w decorators...
        terms_dict.update(poly_terms)
        return terms_dict

    def generate_poly_terms(self):
        """Generate and return polynomial terms for function library.

        Returns:
        poly_terms -- dict containing nonlinear terms for fn lib
        """
        poly_terms = {}
        for i in range(self.poly_deg):
            # Prepare a string descriptor of the polynomial term
            description = self.format_term_descriptor(i+1)
            # Add as a dictionary entry in the Library poly_terms dictionary
            poly_terms[description] = self.data**(i+1)
            # Do the same for the negative polynomial
            description = self.format_term_descriptor(-i-1)
            try:
                data = np.divide(1, self.data**(i+1),
                                 out=np.zeros_like(self.data),
                                 where=self.data != 0)
                poly_terms[description] = data
            except FloatingPointError as err:
                print("Skipped", description, "due to", err)
        return poly_terms
