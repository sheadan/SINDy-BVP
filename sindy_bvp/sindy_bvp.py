"""SINDy-BVP Logic collection."""

# Import Python packages
import pickle
import warnings

# Third-Party Imports
from sklearn.model_selection import train_test_split

# Package Imports
from sindy_bvp.variables import DependentVariable, IndependentVariable
from sindy_bvp.library_builders import TermBuilder, NoiseMaker
from sindy_bvp.differentiators import BaseDifferentiator
from sindy_bvp.optimizers import GroupRegressor, SGTR
from sindy_bvp.analysis.plotter import Plotter
from sindy_bvp.groupers import PointwiseGrouper


class SINDyBVP:
    """Collection of all logic associated with SINDy-BVP."""

    def __init__(self, file_stem: str, num_trials: int,
                 differentiator: BaseDifferentiator,
                 outcome_var: str,
                 noisemaker: NoiseMaker = None,
                 known_vars: list = None,
                 dep_var_name: str = 'u',
                 ind_var_name: str = 'x'):
        """Initialize the attributes from parameters.

        Keyword arguments:
        file_stem --
        num_trials --
        differentiator --
        outcome_var --
        noisemaker --
        known_vars --
        dep_var_name --
        ind_var_name --
        """
        self.file_stem = file_stem
        self.num_trials = num_trials
        self.differentiator = differentiator
        self.outcome_var = outcome_var
        self.noisemaker = noisemaker
        self.known_vars = known_vars
        self.dv_name = dep_var_name
        self.iv_name = ind_var_name

    def sindy_bvp(self, report_results: bool = False):
        """Execute the core logic of SINDy-BVP.

        Keyword argument:
        report_results -- boolean if results should be printed.

        Returns:
        coeffs -- a dictionary containing key-np.ndarray pairs of
        the terms with nonzero coefficients
        pltr - a Plotter instance for generating analysis plots.

        """
        # Step 1. Load in the data
        x_vector, ode_sols, forcings, sl_coeffs = self.load_data()

        # Step 2. Split data into testing and training
        # I've put a warning catch here because train_test_split will give
        # a future warning about test size being complement of train size
        # which is desired in our case and not worth warning the user about
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            data_split = train_test_split(ode_sols, forcings,
                                          train_size=self.num_trials,
                                          random_state=0)

        # Random state is set to 0 for consistency
        sol_train, sol_test, f_train, f_test = data_split

        # Step 3. Build DataFrames containing data from each trial
        dfs = self.build_dataframes(sol_train, f_train)

        # Step 4. Prepare regression
        # Create the group regressor (Uses SGTR regression function above)
        aggregator = PointwiseGrouper(lhs_term=self.outcome_var)
        self.groupreg = GroupRegressor(aggregator, dfs, self.iv_name,
                                       sgtr=SGTR())

        # Step 5. Execute group regression
        self.groupreg.group_regression(known_vars=self.known_vars)
        if report_results:
            self.groupreg.report_learning_results()

        # Step 6. Return learned coefficients and Plotter instance
        # which can generate analysis plots.
        coeffs = self.groupreg.return_coefficients()

        # Construct a plotter object with groupreg (which stores results)
        pltr = Plotter(groupreg=self.groupreg,
                       x_vector=x_vector,
                       true_coeffs=sl_coeffs,
                       dependent_variable=self.dv_name,
                       ode_sols=sol_train,
                       is_sturm_liouville=True,
                       show_legends=False)

        return coeffs, pltr

    def load_data(self):
        """Load in the saved data based on the provided file stem.

        Returns:
        x_vector -- the vector of x values used ([0,0.01,...,10])
        ode_sols -- the solutions to BVP given different forcings
        forcings -- the forcings used for the ode solutions
        sl_coeffs -- the true coefficients used for the BVP model
        """
        x_vector = pickle.load(open(self.file_stem + "x.pickle", "rb"))
        ode_sols = pickle.load(open(self.file_stem + "sols.pickle", "rb"))
        forcings = pickle.load(open(self.file_stem + "fs.pickle", "rb"))
        sl_coeffs = pickle.load(open(self.file_stem + "coeffs.pickle", "rb"))

        return x_vector, ode_sols, forcings, sl_coeffs

    def build_dataframes(self, ode_sols: list, used_fs: list):
        """Build dataframes containing evaluated symbolic functions.

        Keyword arguments:
        ode_sols -- the differential equation solutions to use
        used_fs -- the corresponding forcing functions for the solutions

        Returns:
        dataframes -- a list of dataframes containing the evaluated
        symbolic functions generated from each trial.
        """
        # Create empty lists to house the stacked theta's and u's
        dataframes = []

        # Create the dataframes
        for sol, f in zip(ode_sols, used_fs):
            # Prepare independent variable
            x_data = sol.t
            x = IndependentVariable(self.iv_name, x_data, poly_deg=0)

            # And prepare the dependent variable
            signal = sol.y[0]
            if self.noisemaker is not None:
                signal = self.noisemaker.apply_noise(signal)
            u = DependentVariable(self.dv_name, signal, nonlinear_deg=5)

            # TermBuilder instance builds dataframe from the trial data
            tb = TermBuilder([x], [u], self.differentiator)

            # Inject the forcing function into the dataframe
            tb.register_custom_term("f", f)

            # build_library_terms method computes symbolic functions for all
            # functions in the library and the outcome variable matrices at
            # all spatial coordinates in the system
            dataframe = tb.build_library_terms(lhs_term=self.outcome_var)

            # Add the dataframe to dataframes list
            dataframes.append(dataframe)

        # Return the dataframes
        return dataframes
