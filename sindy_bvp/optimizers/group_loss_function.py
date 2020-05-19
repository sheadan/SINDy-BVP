"""Abstract class to enforce presence of score method."""

# Python imports
from abc import ABC, abstractmethod


class GroupLossFunction(ABC):
    """Abstract class to enforce presence of score method."""

    @abstractmethod
    def score(self, As, bs, xs):
        """Abstract score method.

        Keyword arguments:
        As -- a list of As for Ax=b
        bs -- a list of bs for Ax=b
        xs -- solution(s) to Ax=b
        """
        pass
