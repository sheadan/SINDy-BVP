"""Simple Grouper object for reorganizing and aggregatind data by index.

A list of Pandas DataFrames is provided. The DataFrames must be of the same
length and have the same index entries. The data in the DataFrames are
reorganized by the index, where the n-th row in each DataFrame is collected
and placed into a new DataFrame together. This effectively reorganizes the
data from a list of DataFrames of computed candidate libraries of different
trials to a list of DataFrames of aggregated trials' data at each spatial
position (x_k).
"""

# Python Package Imports
from typing import List

# Package imports
from .base import Grouper

# Third-Party Imports
import pandas as pd


class PointwiseGrouper(Grouper):
    """Reorganize & aggregate data in the DataFrames list by their indices."""

    def group_data(self, dps: List[pd.DataFrame]):
        """Aggregate data by index.

        Keyword Arguments:
        dps -- a list of Pandas DataFrames. Each DataFrame is the computed
        candidate function terms (Theta) where each row is a different x_k
        spatial position. The outcome variable (or lhs_term) is included in
        the input DataFrames.

        Returns:
        grouped_theta -- a list of DataFrames. Each DataFrame is an aggregated
        collection of symbolic functions evaluated at a different spatial pos.
        The index of the list is directly related to the spatial position the
        function is evaluated at.

        grouped_lhs -- similarly, this is a list of DataFrames containining
        only the outcome variable for regression.
        """
        # First, check all the indices are the same
        index_list = [dp.index for dp in dps]
        test_index = index_list.pop()
        compare = [test_index.equals(idx) for idx in index_list]
        assert False not in compare, "Indices of DataFrames not the same!"

        # Compute the group indices
        group_indices = list(test_index)

        # Initialize an empty list to hold the groups
        grouped_theta = []
        grouped_lhs = []

        # Get a list of the terms that will be in theta
        self.theta_terms = list(dps[0].columns)
        self.theta_terms.remove(self.lhs_term)

        # actually assemble the groups:
        for idcs in group_indices:
            # Create dataframe for this group's Theta and U
            lhs = []
            theta = []
            # Extract U and Theta for each datapool
            for datapool in dps:
                # extract the relevant group data for theta
                theta.append(datapool.copy()[self.theta_terms].iloc[idcs])
                # and the left hand side
                lhs.append(datapool.copy()[self.lhs_term].iloc[idcs])

            # If there is data in the temporary 'per_dp' arrays,
            # add it to the grouped_theta, grouped_lhs lists
            if theta:
                grouped_theta.append(pd.concat(theta, axis=1).transpose())
                lhs_dict = {self.lhs_term: lhs}
                grouped_lhs.append(pd.DataFrame.from_dict(lhs_dict))

        return grouped_theta, grouped_lhs
