"""Plotter provides the plotting methods for SINDy-BVP."""

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from .plot_options import PlotOptions


class Plotter(PlotOptions):
    """Collection of methods generating analysis plots."""

    def __init__(self, groupreg,
                 x_vector: np.ndarray,
                 true_coeffs: dict = {},
                 dependent_variable: str = 'u',
                 ode_sols=None,
                 xi_index=None,
                 is_sturm_liouville: bool = True,
                 show_legends: bool = True):
        """Set attributes for the plotting methods.

        Keyword arguments:
        groupreg -- the GroupRegression instance used for regression
        x_vector -- the x vector the data was collected on
        true_coeffs -- the true coefficients used to compare learned
        coefficients against. dictionary containing key-data pairs where
        key is the name of the true coefficient.
        dependent_variable -- string containing the dependent variable name
        ode_sols -- list of ODE solutions, each entry is object returned by
        SciPy solve_ivp method.
        xi_index -- if not using the prospective solution which minimizes PDE
        loss function, this is the index of the prospective solution to plot.
        The default is None, which uses the solution which minimizes PDE loss.
        is_sturm_liouville -- boolean, used for naming coefficients and
        determining if comparison figures should include true coefficients
        show_legends -- boolean for whether or not to place legends on plot
        plot_options -- a collection of options to use for matplotlib. this
        approach should be improved, but works for now. reference plot_options
        from plot_options.py
        """
        # Gather the default plot options from superclass
        super().__init__()

        # Independent and dependent variable names (strings)
        self.independent_var = groupreg.independent_var
        self.dependent_var = dependent_variable

        # By default, assumes the data is not a sturm-liouville problem
        self.is_sturm_liouville = is_sturm_liouville

        # Store the true coefficients and the corresponding x-vector
        self.true_coeffs = true_coeffs  # True S-L coefficients
        self.true_x_vector = x_vector  # And the true x-vector

        # Get regressed coefficients and ind. var. vectors for regressed terms
        self.reg_coeffs = groupreg.return_coefficients()
        self.reg_x_vector = groupreg.iv_vector
        self.lhs_term = groupreg.grouper.lhs_term

        # Boolean for whether or not to show legends
        self.show_legends = show_legends
        self.ode_sols = ode_sols

    def plot_ode_solutions(self, ax: mpl.axes.Axes, number: int = 3,
                           start_idc: int = 0, shift: int = 0):
        """Plot solutions to differential equations used to build dataset.

        Keyword arguments
        ax -- matplotlib.axes.Axes to plot on
        ode_sols -- list of diff eqn solutions from scipy's solve_ivp
        number -- plot only the first <number> solutions in the ode_sols list
        start_idc -- instead of retrieving the first <number> solutions to plot
        start retrieval at the start_idc index of the list
        shift -- provide x-axis offset to each successive solution plotted.
        useful if the solutions overlap significantly.
        """
        # ODE line colors and line properties
        ode_sols = self.ode_sols
        lcolors = self.ode_colors
        lprops = self.ode_opts

        # Plot differential equation solutions
        num_sols_to_plot = min(len(ode_sols), number)
        lines = []
        # Plot each ODE solution
        for i in range(num_sols_to_plot):
            sol = ode_sols[i+start_idc]
            label_u = self.dependent_var + ' solution {}'.format(i+1)
            label_dudx = 'd{}/d{}, solution {}'.format(self.dependent_var,
                                                       self.independent_var,
                                                       i+1)
            if i < len(lcolors):
                line, = ax.plot(sol.t+(i*shift), sol.y[0], color=lcolors[i],
                                linestyle='-', label=label_u, **lprops)
                dline, = ax.plot(sol.t+(i*shift), sol.y[1], color=lcolors[i],
                                 linestyle='--', label=label_dudx, **lprops)
            else:
                line, = ax.plot(sol.t+(i*shift), sol.y[0], linestyle='-',
                                label=label_u, **lprops)
                dline, = ax.plot(sol.t+(i*shift), sol.y[1],
                                 color=line.get_color(), linestyle='--',
                                 label=label_dudx, **lprops)
            lines.append(line)
            lines.append(dline)

        # Format the plot title, x-axis label, y-axis label, and legend
        if self.show_legends:
            leg_lines = [lines[0], lines[1]]
            leg_labels = ['u', '$du/dx$']
            ax.legend(leg_lines, leg_labels, **self.legend_opts)

    def plot_xi(self, ax: mpl.axes.Axes, offset_qty: float = None,
                mean_sub=True):
        """Plot nonzero coefficients for learned terms in Xi matrix.

        Keyword arguments
        ax -- matplotlib.axes.Axes to plot on
        offset_qty -- vertical/ordinate offset for plotting coefficient vectors
        the offset is incremented for every term plotted
        mean_sub -- boolean indicating if data is plotted with mean subtracted
        """
        terms = list(self.reg_coeffs)  # List of terms with nonzero coeffs
        terms.sort()  # Alphabetically sort learned terms

        # Plot the learned coefficients
        offset = 0  # vertical offset
        for i, term in enumerate(terms):
            true_mean = 0

            # If the term is in the "true coefficients", plot the true line
            if self.is_sturm_liouville and term in self.true_model_coeffs:
                if self.true_model_coeffs[term].any():
                    true_mean = np.mean(self.true_model_coeffs[term])
                    data = self.true_model_coeffs[term]
                    data += offset - true_mean
                    ax.plot(self.true_x_vector, data,
                            color=self.coeff_colors[i],
                            label="True ${}$ coefficient".format(term),
                            **self.true_opts)
            # Retrieve and plot the learned term
            data = self.reg_coeffs[term]
            # If have the true mean, use to center data around ~0
            if true_mean and mean_sub:
                data = data - true_mean + offset
            elif mean_sub:
                data = data - np.mean(data) + offset

            ax.plot(self.reg_x_vector, data,
                    marker=self.markers[i],
                    markevery=self.npts,
                    label="Learned ${}$ coefficient".format(term),
                    **self.reg_opts)

            # Compute positive offset (on y axis) to give the line
            if offset_qty is not None:
                offset += offset_qty
            elif self.is_sturm_liouville and term in self.true_model_coeffs:
                offset += max([ceil(max(self.true_model_coeffs[term])), 1])
            else:
                offset += max([ceil(max(self.reg_coeffs[term])), 1])

        if self.is_sturm_liouville:
            for j, term in enumerate(list(self.true_model_coeffs)):
                if term not in terms:
                    if self.true_model_coeffs[term].any():
                        data = self.true_model_coeffs[term]
                        data_mean = np.mean(self.true_model_coeffs[term])
                        data += j + i - data_mean
                        ax.plot(self.true_x_vector, data,
                                color=self.coeff_colors[i],
                                label="True ${}$ coefficient".format(term),
                                **self.true_opts)
                        datamax = ceil(max(self.true_model_coeffs[term]))
                        offset += max([datamax, 1])

        # Format the legend
        if self.show_legends:
            ax.legend(**self.legend_opts)

    def plot_p_and_q(self, ax: mpl.axes.Axes):
        """Plot p(x) and q(x) for Sturm-Liouville models.

        Plots both learned and true coefficients, if true coefficients
        are provided to the Plotter constructor.

        Keyword arguments
        ax -- matplotlib.axes.Axes to plot on
        """
        # Throw an error if using this for a non-Sturm-Liouville operator
        if not self.is_sturm_liouville:
            Exception("This method only applies to Sturm-Liouville operators.")

        # Pull out the parametric coefficient for 'f'
        learned_f_coeff = self.reg_coeffs['f']
        # Compute learned parametric coefficient \phi
        inferred_phi = -1*np.reciprocal(learned_f_coeff)
        # Set NaN entries to 0 (shouldn't be any NaNs though...)
        inferred_phi[np.isnan(inferred_phi)] = 0
        # Save the inferred phi vector
        self.inferred_phi = inferred_phi

        # If it is p(x), call it that on the line label
        if self.lhs_term == 'd^{2}u/dx^{2}':
            phi_label = "Inferred p(x)"
        elif self.lhs_term == 'd^{4}u/dx^{4}':
            phi_label = "Inferred p(x)"
        else:
            phi_label = "Inferred $\phi(x)$"

        # So plot against the true p(x)
        p_x_plotted = self.p_x-np.mean(self.p_x)
        ax.plot(self.true_x_vector, p_x_plotted, color=self.coeff_colors[0],
                label='True $p(x)$', **self.true_opts)
        # for S-L, u_xx (or u_xxxx) regression, phi(x) is p(x)
        ip = inferred_phi-np.mean(self.p_x)
        ax.plot(self.reg_x_vector, ip, marker=self.markers[0],
                markevery=self.npts, label=phi_label, **self.reg_opts)

        # If 'u' is found in the model, there is a q(x) that can be computed
        if 'u' in self.reg_coeffs:
            # Compute q(x)
            inferred_q = self.reg_coeffs['u'] * inferred_phi
            # Plot true and inferred q on a new axis
            offset = max([ceil(max(p_x_plotted)+abs(min(self.q_x))), 1])
            # Plot true and inferred q
            iq = self.q_x-np.mean(self.q_x)+offset
            ax.plot(self.true_x_vector, iq, color=self.coeff_colors[1],
                    label="True $q(x)$", **self.true_opts)
            ax.plot(self.reg_x_vector, inferred_q-np.mean(self.q_x)+offset,
                    marker=self.markers[1], label="Inferred $q(x)$",
                    markevery=self.npts, **self.reg_opts)
            # Save the inferred q(x)
            self.inferred_q = inferred_q

        # Throw a legend on this
        if self.show_legends:
            leglines = ax.get_lines()
            leglabels = [l.get_label() for l in leglines]
            ax.legend(leglines, leglabels, **self.legend_opts)

    def compute_sl_coeffs(self):
        """Compute the Sturm-Liouville coefficients p,q of the system."""
        # Only compute coefficients if system is S-L:
        if not self.is_sturm_liouville or not self.true_coeffs:
            return

        # Compute p(x) for all x in domain (for computing true coefficients)
        self.p_x = self.true_coeffs['p']
        self.px_x = self.true_coeffs['p_x']
        self.q_x = self.true_coeffs['q']

        # Compute the expected observed coefficients (expected entries in Xi)
        self.true_model_coeffs = {'f': -1/self.p_x,
                                  'du/dx': -1*self.px_x/self.p_x,
                                  'u': self.q_x/self.p_x}

        # Determine if the problem is nonlinear
        if 'alpha' in self.true_coeffs and 'nl_exponent' in self.true_coeffs:
            self.is_nonlinear = True
            self.alpha_x = self.true_coeffs['alpha']*np.ones(self.p_x.shape)
            nl_key = 'u^{{{}}}'.format(str(self.true_coeffs['nl_exponent']))
            self.true_model_coeffs[nl_key] = self.alpha_x*self.q_x/self.p_x
        else:
            self.alpha_x = None
            self.is_nonlinear = False

        # Now compute the learned coefficients:
        # Pull out the parametric coefficient for 'f'
        learned_f_coeff = self.reg_coeffs['f']
        # Compute learned parametric coefficient \phi
        inferred_phi = -1*np.reciprocal(learned_f_coeff)
        # Set NaN entries to 0 (shouldn't be any NaNs though...)
        inferred_phi[np.isnan(inferred_phi)] = 0
        # Save the inferred phi vector
        self.inferred_phi = inferred_phi

        # If 'u' is found in the model, there is a q(x) that can be computed
        if 'u' in self.reg_coeffs:
            # Compute q(x)
            self.inferred_q = self.reg_coeffs['u'] * inferred_phi
        else:
            self.inferred_q = np.zeros(self.inferred_phi.shape)

    def generate_analysis_plots(self, save_stem: str = None, num_sols: int = 3,
                                xlims=[0, 10], xi_ylims=None, coeff_ylims=None,
                                plot_xi: bool = False,
                                plot_coeffs: bool = True,
                                plot_sols: bool = True):
        """Generate analysis plots for SINDy-BVP.

        Keyword arguments:
        save_stem -- filename prefix to save the plots
        num_sols -- number of solutions to plot, default 3
        xlims -- the xlimits to use for the analysis plots
        xi_ylims -- matplotlib ylims for xi plot
        coeff_ylims -- matplotlib ylims for coefficients plot
        skip ode plots if no list is provided
        plot_xi -- boolean indicating whether to produce plot of Xi-hat
        plot_coeffs -- boolean indicating to produce plot of coefficients
        plot_sols -- whether to produce plot of ODE solutions used in train
        """
        # Compute the S-L coeffs
        self.compute_sl_coeffs()

        # If provided, make a figure with the ODE solutions
        if plot_sols:
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
            self.plot_ode_solutions(ax)
            self.set_ax_lims(ax, xlims=xlims)  # Set the axis limits, shading
            self.format_ticks(ax)  # Set formatting for the axes' ticks/labels
            if save_stem:
                self.save_figure(save_stem + '-sols')

        if plot_coeffs:
            # Plot for the coefficients
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
            self.plot_p_and_q(ax)
            self.set_ax_lims(ax, xlims=xlims, ylims=coeff_ylims)
            self.format_ticks(ax)
            if save_stem:
                self.save_figure(save_stem + '-pq')

        if plot_xi:
            # Plot for Xi
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
            self.plot_xi(ax)
            self.set_ax_lims(ax, xlims=xlims, ylims=xi_ylims)
            self.format_ticks(ax)
            if save_stem:
                self.save_figure(save_stem + '-xi')

        if plot_sols or plot_coeffs or plot_xi:
            plt.show()

    def format_ticks(self, ax: mpl.axes.Axes):
        """Remove labels and ticks from mpl.axes.

        Keyword arguments:
        ax -- matplotlib.axes.Axes on which to apply changes
        """
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    def save_figure(self, fname: str):
        """Save figure generated with provided filename.

        keyword arguments:
        fname -- filename to save figure. Uses PyPlot interface.
        """
        fname = self.fig_dir + fname + '.svg'
        plt.savefig(fname, dpi=self.dpi, transparent=True)

    def set_ax_lims(self, ax: mpl.axes.Axes, xlims: tuple = None,
                    ylims: tuple = None, yshade: list = None):
        """Set matplotlib axis limits and apply vertical shade.

        Keyword arguments:
        ax -- matplotlib.axes.Axes object to apply changes
        xlims -- tuple for setting x-axis limits (xmin, xmax)
        ylims -- tuple for setting y-axis limits (ymin, ymax)
        yshade -- list of tuples to apply axvspan matplotlib method
        """
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        if yshade is not None:
            for window in yshade:
                ax.axvspan(window[0], window[1], color='grey', alpha=0.75)

    def score_coefficients(self, score_interval=[0.1, 9.9], verbose=False):
        """Print score on coefficient estimates based on L2 norm.

        Keyword arguments:
        score_interval -- window over which to score coefficient estimate
        verbose -- whether to print errors.

        Returns:
        Nothing. Results are printed.
        """
        self.compute_sl_coeffs()
        low_idcs = np.where(self.true_x_vector > score_interval[0])
        high_idcs = np.where(self.true_x_vector < score_interval[1])
        idcs = np.intersect1d(low_idcs, high_idcs)

        p_error = np.linalg.norm(self.inferred_phi[idcs] - self.p_x[idcs])
        p_error = p_error/np.linalg.norm(self.p_x[idcs])
        print('L2 p error: %.4f' % (p_error))

        try:
            q_error = np.linalg.norm(self.inferred_q[idcs] - self.q_x[idcs])
            q_error = q_error/np.linalg.norm(self.q_x[idcs])
            print('L2 q error: %.4f' % (q_error))
        except Exception:
            if verbose:
                print("q(x) error could not be computed.")
                print(Exception)
