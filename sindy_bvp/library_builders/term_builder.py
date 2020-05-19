"""TermBuilder constructs a DataFrame containing all data for regression.

A single 'trial' is provided to TermBuilder, and TermBuilder constructs the
DataFrame from the trial data. The DataFrame contains the numerically evaluated
symbolic functions for SINDy-BVP. The DataFrame contains columns for the
outcome variable and all the functions used in the candidate library Theta.
The rows of the DataFrame correspond to different spatial coordinates at which
each of the candidate functions and outcome variable function(s) are evaluated.
"""

# Standard Library Imports
from typing import List

# Third-Party Imports
import numpy as np
import pandas as pd

# Package Imports
from sindy_bvp.variables import DependentVariable, IndependentVariable
from sindy_bvp.differentiators import BaseDifferentiator as Differentiator


class TermBuilder:
    """Construct DataFrame containing numerically evaluated symbolic funcs."""

    def __init__(self, independent_vars: List[IndependentVariable],
                 dependent_vars: List[DependentVariable],
                 differentiator: Differentiator):
        """Initialize attributes for constructing the DataFrame for regression.

        Keyword parameters:
        independent_vars -- a list of Independent Variables
        dependent_vars -- a list of dependent variables
        differentiator -- a Differentiator instance for numerical
        differentiation of each dependent var. w.r.t. each ind. var.
        """
        # Set keyword parameter attributes
        self.independent_vars = independent_vars
        self.dependent_vars = dependent_vars
        self.differentiator = differentiator
        self.lhs_term = None  # Initialize as 'None'

        # Create a dictionary for custom terms
        self.custom_terms = {}

    def build_library_terms(self, lhs_term: str, skip_cross: bool = False):
        """Construct and return the DataFrame for SINDy-BVP.

        Keyword Parameters:
        lhs_term -- the outcome variable for regression
        skip_cross -- determines whether or not cross-multiplied terms are used

        Returns:
        dframe -- DataFrame for regression
        """
        # First check the variables in the library for dimensions, etc
        self.check_inputs()
        self.lhs_term = lhs_term

        # Create terms from dependent variables
        dv_terms = {}
        for _dv in self.dependent_vars:
            dv_terms.update(_dv.compute_terms())

        # Create terms from independent variables
        iv_terms = {}
        for _iv in self.independent_vars:
            iv_terms.update(_iv.compute_terms())

        # Compute differential terms
        diff_terms = {}
        for _dv in self.dependent_vars:
            for _iv in self.independent_vars:
                differentials = self.differentiator.differentiate(_iv, _dv)
                diff_terms.update(differentials)

        # Now create some cross-terms between specified dictionary sets
        cross_terms = {}
        if not skip_cross:
            terms = self.generate_cross_terms(dv_terms, diff_terms)
            cross_terms.update(terms)

        # Create an empty dictionary for DataFrame data
        data_dict = {}
        term_dicts = [dv_terms, iv_terms,
                      diff_terms, cross_terms,
                      self.custom_terms]
        for term_dict in term_dicts:
            data_dict.update(term_dict)

        # Convert data format to a Pandas DataFrame:
        dframe = pd.DataFrame.from_dict(data_dict)

        # Return the dataframe
        return dframe

    def check_inputs(self):
        """Check the inputs to the TermBuilder prior to constructing DataFrame.

        This method currently checks that the independent_vars and
        dependent_vars lists are not empty and are of length 1. However, other
        checks could/should be implemented such as checking the dimensions of
        the inputs (i.e. compare number and length of independent variables to
        dimensions of dependent variable matrices for multi-dimensional data).
        """
        # Check that IV and DV lists are not empty
        if not self.independent_vars:
            raise Exception("Error - No independent variables provided!")
        if not self.dependent_vars:
            raise Exception("Error - No dependent variables provided!")
        # Check that no more than one IV and DV is provided -- implement later
        if len(self.independent_vars) != 1:
            raise Exception("Error - Too many independent variables defined!")
        if len(self.dependent_vars) != 1:
            raise Exception("Error - Too many dependent variables defined!")
        # Check that IV and DV are same length (while 1D data only)
        iv_data_len = len(self.independent_vars[0].data)
        dv_data_len = len(self.dependent_vars[0].data)
        if iv_data_len != dv_data_len:
            raise Exception("Error - IV-DV data length mismatch!")

    def generate_cross_terms(self, dict_one: dict, dict_two: dict):
        """Compute terms cross-multiplied between two term dictionaries.

        Keyword parameters:
        dict_one -- first dictionary of computed terms
        dict_two -- second dictionary of computed terms

        Adds the cross-multiplied terms to the self.cross_terms attribute.
        """
        cross_terms = {}
        for term_one in dict_one:
            for term_two in dict_two:
                if term_one != self.lhs_term and term_two != self.lhs_term:
                    cross_term_key = str(term_one) + "*" + str(term_two)
                    data = np.multiply(dict_one[term_one], dict_two[term_two])
                    cross_terms[cross_term_key] = data
        return cross_terms

    def register_custom_term(self, term_name: str, term_data: np.ndarray):
        """Provide users ability to register custom terms for regression.

        In practice, this is used to include the forcing function in the
        DataFrame for regression.

        Keyword Parameters:
        term_name: string to describe the term
        term_data: np.ndarray with term data
        """
        # Enforce data length rules:
        if len(term_data) != len(self.independent_vars[0].data):
            err_text = "Term data not same length as dependent var vector!"
            raise Exception(err_text)
        # Register the custom term in custom_terms dict attribute
        self.custom_terms[term_name] = term_data
