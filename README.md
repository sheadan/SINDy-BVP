# SINDy-BVP: Sparse Identification of Nonlinear Dynamics for Boundary Value Problems

## Paper Reference

This will be updated with hyperlinks when the paper is published.

### Repository Overview

This repository contains the following directories:

- **Base directory**: Contains Jupyter notebooks that exemplify how to use SINDy-BVP ('Fig' prefixes) and that generate and save the data used for analysis ('System #' prefixes).
- **sindy_bvp**: Contains the SINDy-BVP package files, which are further described below.
- **data**: Contains data for analysis in the saved '.pickle' files. These are generated from the 'System' notebooks in the base directory.
- **Figs**: Contains scalable vector graphics (SVG) images for all of the figure components in the accompanied paper.
- **Supporting Information**: Contains notebooks and figure components for additional concepts and ideas (e.g. what if only one trial experiment is used for regression?)

### SINDy-BVP Package
First, I will list the subpackages and each of their contents. Each subpackage is its own directory (this is typical for Python). After describing the subpackages, I will describe the typical use of the code as exemplified by the 'System' and 'Fig' notebooks in the base directory.

#### Subpackages

1. **analysis**: Contains plotting logic (plotter.py) and default options (plot_options.py). Plotting is done with [matplotlib](https://matplotlib.org/).
2. **bvp_solver**: Contains a list of equations used to define the differential equation models (equations.py), a randomized initial conditions generator (ic_generator.py), a simple wrapper for SciPy's solve_ivp initial value problem solver (ivp_solver.py), and a BVP solver which encapsulates the IVP solver (bvp_solver.py). The BVP solver implements a simple shooting method which models a technique from [Data-Driven Modeling & Scientific Computation: Methods for Complex Systems & Big Data (Kutz, 2013)](https://amath.washington.edu/research/publications/data-driven-modeling-scientific-computation-methods-complex-systems-big-data).
3. **differentiators**: Contains an abstract base class for a differentiator (base_differentiator.py), a finite differences class (finite_differences.py), and a polynomial interpolation method (poly_interp.py). Both methods are heavily based on Sam Rudy's code for [Parametric PDE FIND](https://github.com/snagcliffs/parametric-discovery). The polynomial interpolation fits data in a window to a Chebyshev polynomial with SciPy, then takes derivatives. A window of a specified width is used around each data point. The FD method is central second order.
4. **groupers**: Contains a base class (base.py) and implemented pointwise-spatial grouper (pointwise_grouper.py). These "grouper" methods are designed to organize a collection of trials for regression. Typically, you have \(m\) sets of data \({**U**_i, **F**_i}\) , each of which is discretized over \\(n\\) spatial points. The **library_builders** subpackage takes each set of data and generates data matrices of the numerically evaluated terms from the data. The **groupers** package then takes the data matrices constructed by the **library_builders** from the \\(m\\) data sets and organizes them for the regression described in the paper.
5. **library_builders**: Contains a TermBuilder (term_builder.py) and a noise generator (noise_maker.py). The term_builder takes a *single* trial \\({**U**_i, **F**_i}\\) and creates a numerically evaluated term library \\(**\Theta**\\) and the outcome variable (e.g. the second spatial derivative of **U**) for the data in that trial. The noise generator applies a gaussian white noise to the data in **U** prior to sending the signal to the TermBuilder. It can also apply a smoothing filter to emulate experimental data treatment.
6. **optimizers**:  Contains an abstract base class for the group regression loss function (group_loss_function.py) and the group regression optimizer (optimizer.py). The generalized SGTR algorithm is implemented in SGTR.py, the PDE loss function is in pde_loss_function.py, and ridge regression is found in ridge.py. The GroupRegressor class found in group_reg.py contains all the generalized logic for executing these pieces together, though in principal you could use different components (e.g. lasso instead of ridge).
7. **variables**: Handy definitions for defining a 'dependent' and 'independent' variable. These definitions allow the TermBuilder in library_builders to figure out which variables should be differentiated and with respect to which other variables. It's also useful for implementing data dimensionality checks. Each of the definitions have utility methods for generating exponent functions of the variable (i.e. polynomials for independent variables and nonlinear terms for dependent variables).

#### Using the Package

The file **sindy_bvp.py** can be found in the base directory. This is where the logic tying together all the subpackages is contained. The idea here is to simplify the usage of SINDy-BVP (the method) to a few commands. First, a SINDyBVP object is constructed using the constructor method which asks for the following:

    file_stem		# File stem for data files in ./data/ directory with u and f data
    num_trials      # Number of trials to randomly sub-sample from the data file
    differentiator  # An instance of a differentiator object to use for differentiation
    outcome_var 	# The name of the outcome variable (e.g. "d^{2}u/dx^{2}" for u_xx)
    noisemaker		# An instance of a noisemaker object, if noise is to be applied
    known_vars		# The variables of the known operator, *if* your operator is known
    dep_var_name	# Purely for nomenclature, the string name of the dep. var. (e.g. "x")
    ind_var_name	# Similarly but for independent var (e.g. "u")
The SINDyBVP.sindy_bvp() method is then called. This method loads the relevant data, randomly selects trial data to use for regression (i.e. finding the model and parameters), and sends that data to a TermBuilder object to construct [Pandas](https://pandas.pydata.org/) [DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) containing symbolic functions. Each trial is sent individually to the TermBuilder, which creates a DataFrame from each individual trial. The DataFrame has \\(p+1\\) columns, where \\(p\\) is the number of candidate functions to use for SINDy regression. The additional column corresponds to the regression outcome variable. After a DataFrame has been constructed for each of the trials, the data is sent to a grouper object, which takes the DataFrames and reorganizes them for regression, as described in the paper. The organized data is used by the GroupRegressor object to perform the SGTR algorithm. The results are reported, and the coefficients are computed for the operator L (rather than for the algebraically manipulated form learned by SINDy-BVP). The coefficients are returned with a configured plotter object, which can be used to visualize the results.

This process flow can be seen by inspecting any of the 'Figure' Jupyter notebooks in the repository base directory.

#### Generating the Data

The above section assumes data has already been generated and is ready for use. The 'System' Jupyter notebooks in the base directory show examples of how to generate the data using the BVP shooting method which solves an initial value problem with the Runge-Kutta method, compares the right-hand-side boundary value to the desired boundary condition, and adjusts the left-hand-side boundary value(s) to try to meet the right-hand-side boundary values while maintaining set boundary conditions. The notebooks walk through every step pretty explicitly and only the shooting method and IVP solver logic are hidden away in the subpackages.

#### Tweaking Hyperparameters

If you want to change the hyperparameters used in this work (i.e. epsilon=1e-6, lambda=1e-5), there is some minor modification you will need to perform. Both of these parameters, and others, are set by the SGTR class, which is currently hard-coded to use defaults. It is not too challenging to change this; you could modify the sindy_bvp() method to accept an SGTR object parameter, which is used in the constructor for the GroupRegressor. This is currently line 84 of sindy_bvp.py. The SGTR object can be configured to use a different parameter epsilon (by passing a differently-configured PDELossFunction object to the constructor), a different lambda (by passing a differently-configured Optimizer/Ridge object to the constructor), or use different threshold functions, optimizers, and loss functions altogether. This is not meant to replace a full guide to custom-configuration of the method, but as a starting point for anyone curious to re-use this code.



#### Final Thoughts

A lot of the logic is glossed over here, please feel free to contact me directly or open an issue!


### Changelog
2020-05-19: Initial commit. I will update this Readme with an overview of the package, a description of how it works, and reference to the research paper.

2020-05-20: Initial write-up of repository contents and basic package usage.

