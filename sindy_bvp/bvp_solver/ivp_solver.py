"""Initial Value Problem solver class built around Scipy's solve_ivp."""

# Third party imports
import numpy as np
from scipy.integrate import solve_ivp


class IVPSolver:
    """Class wraps solve_ivp to compute multiple solutions."""

    def __init__(self, ode_method='RK45', t_min=0, t_max=1,
                 dt=0.01, rtol=1e-9, atol=1e-10):
        """Initialize attributes passed as parameters to solve_ivp method.

        Keyword arguments:
        ode_method -- string passed to solve_ivp indicating method parameter
        t_min -- the first entry for t_span tuple parameter
        t_max -- the second entry for t_span tuple parameter
        dt -- used to compute the t_eval parameter so all solutions are on
        the same grid.
        rtol -- passed to solve_ivp as rtol parameter
        atol -- passed to solve_ivp as atol parameter
        """
        self.ode_method = ode_method

        self.rtol = rtol
        self.atol = atol

        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt

        self.t_eval = np.linspace(t_min, t_max, int((t_max-t_min)/dt)+1)

    def solve_ode_ivp(self, ode_func: callable, init_vals: list,
                      shift_eval: float = 0.0):
        """Solve the IVP given initial values.

        Keyword arguments:
        ode_func -- the differential equation definition used for solve_ivp
        init_vals -- initial values
        shift_eval -- amount to shift the solution domain, default 0.0

        Returns:
        sol -- solution computed by solve_ivp
        """
        # Solve and return the IVP
        return solve_ivp(ode_func,
                         [self.t_min+shift_eval, self.t_max+shift_eval],
                         init_vals,
                         t_eval=self.t_eval + shift_eval,
                         method=self.ode_method,
                         rtol=self.rtol,
                         atol=self.atol)

    def get_multiple_sols(self, ode_func: callable, init_vals_list: list,
                          shift_eval: float = 0.0):
        """Compute multiple solutions from a list of multiple initial values.

        This method iterates through init_vals_list, which is a list of ICs,
        and computes a solution to the provided differential equation ode_func
        provided each initial condition.

        Keyword arguments:
        ode_func -- the differential equation definition used by solve_ivp.
        init_vals_list -- a list of initial values lists to use for solve_ivp.
        shift_eval -- the amount to shift solutions as they are computed.

        Returns:
        solutions -- a list of solutions corresponding to the provided ICs
        """
        # Prepare a list for the ODE solutions
        solutions = []

        # for each entry in the init_vals_list, solve the ode and store results
        for i, init_vals in enumerate(init_vals_list):
            sol = self.solve_ode_ivp(ode_func, init_vals, shift_eval)
            # Store it in the solutions list
            solutions.append(sol)

        # Return the solutions list
        return solutions
