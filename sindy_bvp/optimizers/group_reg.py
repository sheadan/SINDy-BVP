"""GroupRegressor organizes logic associated with SINDy-BVP regression."""

# Python Package Imports
from typing import List

# Third-Party Imports
import numpy as np
import pandas as pd

# Package imports
from sindy_bvp.groupers import Grouper
from .SGTR import SGTR


class GroupRegressor:
    """Logic collection for executing SINDy-BVP regression."""

    def __init__(self,
                 grouper: Grouper,
                 datapools: List[pd.DataFrame],
                 independent_var: str,
                 sgtr: SGTR = None):
        """Initialize attributes for regression.

        Keyword arguments:
        regfunc -- the optimization function used to solve Ax=b
        grouper -- the algorithm which re-organizes data for regression.
        the grouper.group_data() should return a list of As and bs to solve
        Ax=b. In SINDy-BVP, Ax=b is solved at each spatial point, so each
        A in As and b in bs correspond to a single spatial point.
        datapools -- a list of DataFrames for each trial
        independent_var -- string name of the independent variable
        """
        self.grouper = grouper
        self.independent_var = independent_var

        # Drop rows with NaN in datapools (NaNs from numerical differentiation)
        self.datapools = [datapool.dropna().reset_index(drop=True)
                          for datapool in datapools]

        # Confirm that the independent variable is in the datapools
        assert(independent_var in datapools[0].columns)

        # Initialize an attribute for independent variable vector
        # This is useful for plotting results, esp. if the datapool was trimmed
        # by the dropna() method
        self.iv_vector = self.datapools[0][self.independent_var]

        # Set SGTR attribute
        self.sgtr = sgtr
        if sgtr is None:
            self.sgtr = SGTR()

    def group_regression(self, known_vars: List[str] = None):
        """Execute group regression on Ax=b list.

        Keyword arguments:
        known_vars -- list of variables known to exist in the solution.

        Returns:
        nothing

        This saves the Xi, Losses, and Tols used by the SGTRidge algorithm.
        """
        # Grab a copy of the datapools to work with for regression.
        dps = [dp.copy() for dp in self.datapools]

        # Save the terms used in for regression (for reporting results and
        # ungrouping the coefficients)
        self.reg_terms = list(self.datapools[0].columns)

        # Pull out the 'known variables' of the equation, if they are known
        if known_vars is not None:
            for var in known_vars:
                assert(var in self.datapools[0].columns)
            if self.grouper.lhs_term not in known_vars:
                known_vars.append(self.grouper.lhs_term)
            dps = [dp[known_vars] for dp in self.datapools]
            self.reg_terms = known_vars
            self.sgtr.num_tols = 1
        self.reg_terms.remove(self.grouper.lhs_term)

        # Use the grouping algorithm to group the data for regression inputs
        As, bs = self.grouper.group_data(dps)

        # Save a copy of the As and bs for the Ax=b regressions as attribute
        self.As = As
        self.bs = bs

        # Regression executed by SGTR algorithm.
        xs, tols, losses = self.sgtr.optimize(As, bs)

        # It's best to record only unique solutions and the corresponding
        # losses and tolerances
        self.Xi, self.Tols, self.Losses = self.find_unique_xs(xs, tols, losses)

    def find_unique_xs(self, xs, tols, losses):
        """Find unique xs and the corresponding tolerance and Losses.

        Determines which solutions are unique by inspecting the loss
        function values. If the exact same loss is computed, the solutions
        are assumed to be similar.

        Keyword arguments:
        xs -- a list of xs from the regression function using different
        threshold tolerances for iterative thresholding.
        tols -- a list of tolerances used to compute each x
        losses -- a list of PDE loss function losses for each solution

        Returns:
        unique_xs -- a list of the unique xs, based on the loss
        unique_tols -- a list of unique tolerances, based on loss
        unique_losses -- a list of tolerances
        """
        unique_xs = []
        unique_tols = []
        unique_losses = []

        for i in range(len(xs)):
            if losses[i] not in unique_losses:
                unique_xs.append(xs[i])
                unique_tols.append(tols[i])
                unique_losses.append(losses[i])

        return unique_xs, unique_tols, unique_losses

    def ungroup_data(self, coeff_data):
        """Compute coefficients' values at each spatial coordinate.

        Keyword arguments:
        coeff_data -- the coefficient data with which to compute

        Returns:
        new_data -- the coefficient values organized as a dictionary
        with keys indicating the function the coefficient comes before
        and the entries being Numpy data vectors with the values of the
        coefficients.
        """
        # Find the length of the actively used datapools
        data_length = len(self.datapools[0].index)

        # Create an empty array of appropriate length to house the values
        new_data = np.zeros(data_length)

        for i, idcs in enumerate(self.grouper.group_indices):
            # Determine which indices the value will be assigned to
            low_idx = idcs[0]
            upp_idx = idcs[1]+1
            if i == len(self.group_indices)-1:
                upp_idx = data_length
            # And assign the new value to those indices
            new_data[low_idx:upp_idx] = coeff_data[i]

        return new_data

    def report_learning_results(self, report_number: int = None,
                                show_possible: bool = True,
                                verbose: bool = True):
        """Print the terms selected by SINDy-BVP.

        Keyword arguments:
        report_number -- the number of additional solutions to print. Default
        shows solution that minimizes the PDE loss function. This parameter
        allows additional possible solutions to be printed.
        show_possible -- boolean indicating if the algorithm should print all
        possible results in addition to the report_number solution.
        verbose -- boolean indicating if additional information should be
        printed including the value of the loss function and the mean of the
        coefficient vectors.
        """
        theta_terms = self.reg_terms

        def print_coeffs(coeffs: dict):
            """Print out additional detail about a solution x."""
            print("Selected: ", len(coeffs), " term(s).")
            if len(coeffs) <= 10:
                for term in coeffs:
                    l2_norm = np.linalg.norm(coeffs[term])
                    mean = np.mean(coeffs[term])
                    std_dev = np.std(coeffs[term])
                    print(term, ": ", l2_norm, mean, std_dev)

        # Get coefficients for the minimum loss
        coeffs = self.return_coefficients()

        # Print the results:
        if show_possible:
            print("Possible terms:\n{}\n".format(theta_terms))

        if verbose:
            print("PDE Find Method:")
            print("Minimum Error: ", min(self.Losses))
            print_coeffs(coeffs)
        else:
            print("Model includes:", list(coeffs))

        if report_number is not None:
            print("\n")
            print("Other possible results:")
            for i in range(report_number):
                coeffs = self.return_coefficients(-1*i)
                print_coeffs(coeffs)

    def get_coefficients(self, x, threshold: float = 10**-10):
        """Return the coefficients in x that are nonzero.

        Keyword arguments:
        x -- the solution to compute the coefficient from.
        threshold -- the threshold to consider a coefficient 'nonzero'
        based on the L2 norm of the coefficient vector.
        """
        # Initialize the coeffs dictionary
        coeffs = {}
        # Find coefficients with norm greater than threshold
        for j in range(x.shape[0]):
            if np.linalg.norm(x[j, :]) > threshold:
                term = self.reg_terms[j]
                coeffs[term] = x[j, :]
        return coeffs

    def return_coefficients(self, n: int = None):
        """Return the coefficients from a specific solution x.

        Keyword arguments:
        n -- integer index of solution to return coefficients from.
        by default this is None, which then uses the solution that
        minimizes the PDE loss function.
        """
        if n is None:
            n = np.argmin(self.Losses)
        if n > len(self.Xi):
            n = np.argmin(self.Losses)
        x = self.Xi[n]
        return self.get_coefficients(x)
