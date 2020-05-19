"""PDE Loss Function class."""

import numpy as np
from numpy.linalg import norm
from .group_loss_function import GroupLossFunction


class PDELossFunction(GroupLossFunction):
    """PDE Loss Function for computing loss of SGTR algorithm."""

    def __init__(self, epsilon=1e-6):
        """Initialize epsilon parameter."""
        self.epsilon = epsilon

    def score(self, As, bs, x):
        """Compute score of solution x given list of As and bs.

        Keyword arguments:
        As -- list of As
        bs -- list of bs
        x -- x solution, stacked, where len(hstack(bs)) == len(x)

        Returns:
        score -- PDE loss value for solution x to grouped Ax=b.
        """
        epsilon = self.epsilon
        D, m = x.shape
        n, _ = As[0].shape
        N = n*m
        rss_list = []
        for j in range(m):
            rss = norm(bs[j] - As[j].dot(x[:, j].reshape(D, 1)), 2)
            rss_list.append(rss**2)
        rss = np.sum(rss_list)
        k = np.count_nonzero(x)/m
        loss = N*np.log(rss/N + epsilon) + 2*k + (2*k**2+2*k)/(N-k-1)

        return loss
