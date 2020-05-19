"""Ridge regression computed analytically."""

import numpy as np
from .optimizer import Optimizer


class Ridge(Optimizer):
    """Ridge regression for solving Ax=b."""

    def __init__(self, lambda_: float):
        """Initialize lambda regularization constant."""
        self.lambda_ = lambda_

    def optimize(self, A, b):
        """Compute solution to ridge regression analytically.

        Keyword arguments:
        A -- A in Ax=b
        b -- b in Ax=b

        Returns:
        x -- solution to Ax=b.
        """
        lam = self.lambda_
        if lam != 0:
            x = np.linalg.solve(A.T.dot(A)+lam*np.eye(A.shape[1]), A.T.dot(b))
        else:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        return x
