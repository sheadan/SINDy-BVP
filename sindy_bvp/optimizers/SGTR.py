"""SGTR algorithm for iteratively solving Ax=b over multiple A's and b's."""

import numpy as np
from numpy.linalg import norm as Norm
import pandas as pd

from .optimizer import Optimizer
from .ridge import Ridge
from .group_loss_function import GroupLossFunction
from .pde_loss_function import PDELossFunction


class SGTR:
    """Class containing logic for the SGTR algorithm."""

    def __init__(self,
                 point_optimizer: Optimizer = Ridge(lambda_ = 1e-5),
                 loss_func: GroupLossFunction = PDELossFunction(),
                 threshold_func: callable = Norm,
                 num_tols: int = 50,
                 normalize_by: int = 2):
        """Initialize components of the SGTR algorithm.

        Keyword arguments:
        point_optimizer -- the solver for a single Ax=b problem (Ridge in SGTR)
        loss_func -- The loss function used for grading prospective solutions.
        threshold_func -- the function used for thresholding
        num_tols -- number of threshold tolerances to try for iterative thresh.
        normalize_by -- the norm by which to normalize the cols in As and bs
        """
        self.point_optimizer = point_optimizer.optimize
        self.loss_func = loss_func.score
        self.threshold_func = threshold_func
        self.num_tols = num_tols
        self.normalize = normalize_by

    def format_inputs(self, As, bs):
        """Format As and bs to list of ndarrays.

        Keyword arguments:
        As -- list of As
        bs -- list of bs

        Returns:
        As -- list of As as a list of ndarrays
        bs -- list of bs as a list of ndarrays
        """
        As = [self.convert_to_ndarray(A.copy()) for A in As]
        bs = [self.convert_to_ndarray(b.copy()) for b in bs]
        return As, bs

    def convert_to_ndarray(self, array_like):
        """Convert an ndarray-like object to an ndarray.

        Keyword arguments:
        array_like -- an ndarray-like object

        Returns:
        ndarray -- object converted to ndarray
        """
        if type(array_like) == pd.DataFrame:
            return array_like.values
        else:
            try:
                return np.asarray(array_like)
            except Exception as err:
                print("Exception on convering data:")
                print(err)
                raise(Exception("can't convert data to numpy array!"))

    def compute_norms(self, As, bs):
        """Compute the norms of As and bs. As list is computed column-wise.

        Keyword argument:
        As -- list of As
        bs -- list of bs

        Returns:
        As_norms -- list of As norms
        bs_norms -- list of bs norms

        The norm computed is based on the attribute self.normalize. Note that
        As_norms is computed by taking all As, stacking them, and then
        computing the norm of each column.
        """
        m = len(As)  # m is the number of individual optimizations to run
        # in SINDy-BVP, m is the number of spatial positions

        n, d = As[0].shape  # d is the number of candidate functions
        # and n is the number of trials

        # Initialize an empty vector to hold the norm of each candidate
        # function. the norm is evaluated over ALL spatial positions for
        # each candidate function.
        As_norms = np.zeros(d)
        for i in range(d):
            data = np.hstack([A[:, i] for A in As])
            As_norms[i] = Norm(data, self.normalize)

        # Now normalize the bs
        bs_norms = [m*Norm(b, self.normalize) for b in bs]

        return As_norms, bs_norms

    def normalize_data(self, As, bs, As_norms, bs_norms):
        """Normalize the data in As and bs by norms As_norms, bs_norms.

        Keyword arguments:
        As -- list of As
        bs -- list of bs
        As_norms -- list of As norms
        bs_norms -- list of bs norms

        Returns:
        normalized_As -- As normalized by the As_norms
        normalized_bs -- bs normalized by the bs_norms
        """
        normalized_As = [A.copy() for A in As]
        normalized_bs = [b.copy() for b in bs]
        for i in range(len(As)):
            normalized_As[i] = As[i].dot(np.diag(As_norms**-1))
            normalized_bs[i] = bs[i]/bs_norms[i]

        return normalized_As, normalized_bs

    def compute_tolerances(self, As, bs):
        """Compute the range of tolerances to use for iterative thresholding.

        Keyword arguments:
        As -- list of As
        bs -- list of bs

        Returns:
        tols -- range of tolerances to use for iterative thresholding.
        """
        # Compute the range of tolerances to use for thresholding
        opt = self.point_optimizer  # Use shortcut for optimizer
        # Compute the solution x for each group using ridge regression
        x_ridges = [opt(A, b) for (A, b) in zip(As, bs)]
        # Stack the solutions into matrix, so that each column contains
        # the coefficient vector for a single candidate function, where
        # each row is a single spatial point.
        x_ridge = np.hstack(x_ridges)
        # Get the norm for each of the candidate function coefficient vectors
        xr_norms = [Norm(x_ridge[j, :]) for j in range(x_ridge.shape[0])]
        # Determine the maximum of these norms
        max_tol = np.max(xr_norms)
        # And the minimum
        min_tol = np.min([x for x in xr_norms if x != 0])
        # And compute a range of tolerances to use for thresholding
        tolerance_space = np.linspace(np.log(min_tol), np.log(max_tol),
                                      self.num_tols)
        tols = [0]+[np.exp(alpha) for alpha in tolerance_space][:-1]
        # return the tolerances
        return tols

    def optimize(self, As, bs):
        """Execute SGTR algorithm.

        Inputs:
        As -- list of As
        bs -- list of bs

        Returns:
        xs -- all prospective solutions produced by iter. thresh.
        tols -- tolerances used for iterative thresholding
        losses -- the losses computed by loss function (typ. PDE Loss Fn)
        """
        if len(As) != len(bs):
            raise Exception('Number of Xs and ys mismatch')
        As, bs = self.format_inputs(As, bs)

        np.random.seed(0)

        if isinstance(self.normalize, int):
            As_norms, bs_norms = self.compute_norms(As, bs)
            As, bs = self.normalize_data(As, bs, As_norms, bs_norms)

        tols = self.compute_tolerances(As, bs)

        # Execute SGTR for each thresholding tolerance
        xs = []
        losses = []

        for i, tol in enumerate(tols):
            try:
                x = self.iterative_thresholding(As, bs, tol=tol)
                xs.append(x)
                loss = self.loss_func(As, bs, x)
                losses.append(loss)
            except Exception as exc:
                print(exc)
                pass

        if isinstance(self.normalize, int):
            xs = self.scale_solutions(As, bs, xs, As_norms, bs_norms)

        return xs, tols, losses

    def iterative_thresholding(self, As, bs, tol: float, maxit: int = 10):
        """Iterate through tolerances for thresholding and produce solutions.

        Keyword arguments:
        As -- list of As for Ax=b
        bs -- list of bs for Ax=b
        tol -- the tolerance to use for thresholding
        maxit -- the maximum number of times to iteratively threshold

        Returns:
        W -- final solution to iterative thresholding at tolerance tol.
        """
        # Assign shorter alias for point-wise optimizer
        opt = self.point_optimizer
        tfunc = self.threshold_func

        # Define n, d, m
        n, d = As[0].shape  # n is num of trials. d is num of candidate funcs
        m = len(As)  # m is the number of spatial positions.

        # Get initial estimates
        W = np.hstack([opt(A, b) for [A, b] in zip(As, bs)])
        num_relevant = As[0].shape[1]  # assume all candidate functions matter

        # Select indices of candidate functions exceeding the threshold
        # as graded by the thresholding function
        biginds = [i for i in range(d) if tfunc(W[i, :]) > tol]

        # Execute iterative tresholding
        for j in range(maxit):
            # Figure out which items to cut out
            smallinds = [i for i in range(d) if Norm(W[i, :]) < tol]
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds):
                j = maxit-1
            else:
                num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0:
                    print("Tolerance too high - all coefficients thresholded.")
                break
            biginds = new_biginds

            # Otherwise get a new guess
            for i in smallinds:
                # zero out thresholded columns
                W[i, :] = np.zeros(len(As))
            if j != maxit - 1:
                for i in range(m):
                    x = opt(As[i][:, biginds], bs[i])
                    x = x.reshape(len(biginds))
                    W[biginds, i] = x
            else:
                # Get final least squares estimate
                for i in range(m):
                    r = len(biginds)
                    W[biginds, i] = np.linalg.lstsq(As[i][:, biginds],
                                                    bs[i],
                                                    rcond=None)[0].reshape(r)

        return W

    def scale_solutions(self, As, bs, xs, As_norms, bs_norms):
        """Scale solutions back based on norms.

        Keyword arguments:
        As -- list of As
        bs -- list of bs
        xs -- list of prospective solutions from iterative thresholding
        As_norms -- norm of As
        bs_norms -- norm of bs

        Returns:
        xs -- re-scaled solutions based on As and bs norms.
        """
        for x in xs:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x[i, j] = x[i, j]*bs_norms[j]/(As_norms[i])

        return xs
