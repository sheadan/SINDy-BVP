"""Meta class definition (interface) for Grouper classes.

Base class defines __init__ and abstract group_data methods. The __init__
stores the 'lhs_term' attribute, which is the outcome variable or
'left hand side' of the SINDy regression. The group_data abstract method
is a required method for any Grouper class.
"""

# Python Package Imports
from typing import List
from abc import ABC, abstractmethod

# Third-Party Imports
import pandas as pd


class Grouper(ABC):
    """Define meta class for Grouper objects."""

    def __init__(self, lhs_term: str):
        """Define 'lhs_term' attribute.

        Keyword args:
        lhs_term -- string describing the outcome variable for SINDy-BVP
        """
        self.lhs_term = lhs_term

    @abstractmethod
    def group_data(self, dps: List[pd.DataFrame]):
        """Enforce subclasses must implement this function."""
        pass
