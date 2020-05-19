"""Boundary Value Problem solver based on the shooting method."""

# Package imports
from .ivp_solver import IVPSolver


class BVPShooter():
    """Class implements shooting method to compute BVP solutions."""

    def __init__(self, ivp_solver: IVPSolver, bc_index: int = 0,
                 ic_index: int = 0, attempts: int = 25,
                 bv_tolerance: float = 1e-3):
        """Initialize attributes for solving BVPs.

        This algorithm is currently configured to only check Dirichlet or
        Neumann type boundary conditions. Mixed conditions can also be solved
        if the right-side ('second') boundary is constrained by a Dirichlet or
        Neumann type condition (y or y').

        Keyword arguments:
        ivp_solver -- IVPSolver instance for computing IVP solutions
        bc_index -- index of boundary value to check (0=y, 1=y', 2=y'', etc)
        ic_index -- index of initial value to modify (0=y, 1=y', 2=y'', etc)
        attempts -- Number of times to try shooting
        bv_tolerance -- target tolerance for boundary value
        """
        self.ivp_solver = ivp_solver
        self.bc_index = bc_index
        self.ic_index = ic_index
        self.attempts = attempts
        self.target_tol = bv_tolerance

    def shoot(self, ode_func: callable, ic_list: list, target_bv: float = 0,
              increment: float = None):
        """
        Execute the shooting method to attempt to solve a BVP.

        Keyword arguments:
        ode_func -- the differential equation definition expected by solve_ivp
        ic_list -- the initial conditions to start shooting with
        target_bv -- the target boundary value for the self.bc_index condition
        increment -- amount to increment (or decrement) the initial condition
        value between attempts.

        Returns:
        final_sol -- the final solution obtained using the shooting method
        """
        # Confirm that BC and IC indices are valid
        assert self.bc_index in range(len(ic_list))
        assert self.ic_index in range(len(ic_list))

        # Assume a value for the increment if none given
        if increment is None:
            increment = 0.5*ic_list[self.ic_index]

        # Use shooting method to solve the problem
        for i in range(self.attempts):
            # Solve the DiffEq as an IVP using provided initial values
            sol = self.ivp_solver.solve_ode_ivp(ode_func, ic_list)

            # Current boundary value
            current_bv = sol.y[self.bc_index][-1]

            # If you've converged, break out of the loop
            if abs(target_bv - current_bv) < self.target_tol:
                break

            # Track previous guess's initial condition
            prev_guess = ic_list[self.ic_index]

            # Adjust the ICs if the guess was wrong
            if current_bv < target_bv:
                # Increase IC if the guess result is lower than target value
                ic_list[self.ic_index] += increment
            else:
                # Decrease IC guess if result is higher than the target value
                ic_list[self.ic_index] += -1*increment
                # And decrease the increment size
                increment = increment/2

            # Now check that you're actually adjusting your guess within
            # machine precision. If not, or you're on the last loop, then
            # the shooting method has failed to converge on a solution
            if ic_list[self.ic_index] == prev_guess or i == self.attempts - 1:
                # If you haven't converged, change the status and message of
                # the solution to reflect that.
                sol.status = 2
                sol.message = "Failed to converge on target boundary values."
                break

        final_sol = sol

        return final_sol

    def generate_multiple_experiments(self, diffeq, forcings: list,
                                      init_vals_list: list,
                                      verbose: bool = True):
        """Generate solutions to a specific BVP subjected to different forcings.

        Keyword arguments:
        diff_eq -- a differential equation definition expecting params (x,y,f)
        where x is the independent variable, y is the dependent, and f is an
        applied forcing function
        forcings -- a list of lambda functions defining forcing functions
        init_vals_list -- a list of initial values used to compute solutions.
        the init_vals_list are all starting points, as shooting method changes
        the initial conditions as needed to converge on a solution which
        satisfies the constraints at both boundaries.
        verbose -- a boolean indicating if solutions statuses should be printed

        Returns:
        ode_sols -- a list of solutions to the provided differential equation
        used_fs -- a list of the forcing functions used for each solution

        Note len(ode_sols) == len(used_fs) where only converged solutions and
        the corresponding forcing function are added to the returned lists.
        """
        ode_sols = []  # A list to hold BVP solutions
        used_fs = []  # A list holding the used forcing functions
        # Inform the observer what is happening during BVP shooting
        if verbose:
            description = "0 status = complete, 2 status = failed to converge"
            description += "\nSolution statuses:"
            print(description, end=" ")
        # Generate a solution for each f in forcings list
        for f in forcings:
            for init_vals in init_vals_list:
                sol = self.shoot(lambda x, y: diffeq(x, y, f=f(x)),
                                 init_vals, 
                                 increment=0.25)
                if sol.status == 0:
                    ode_sols.append(sol)
                    used_fs.append(f)
                if verbose:
                    print(sol.status, end=" ")
        # Report number of solutions generated.
        if verbose:
            print("\nCreated", len(ode_sols), "solutions.")
        # Return the ode_sols and used_fs
        return ode_sols, used_fs
