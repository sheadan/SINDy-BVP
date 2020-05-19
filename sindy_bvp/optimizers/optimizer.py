"""Abstract class for a generic optimizer that solves Ax=b."""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Abstract optimizer class requiring optimize method."""

    @abstractmethod
    def optimize(self, A, b):
        """Optimize method required by this ABC.

        Keyword arguments:
        A -- A in Ax=b
        b -- b in Ax=b

        Returns:
        x -- x in Ax=b, the best fit solution based on
        the optimization method.
        """
        pass
