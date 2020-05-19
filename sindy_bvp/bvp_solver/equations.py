"""Equation definitions for SINDy-BVP."""

# Python Package Imports
from typing import List

# Third party imports
import numpy as np


def sturm_liouville_function(x, y, p, p_x, q, f, alpha=0, nonlinear_exp=2):
    """Second order Sturm-Liouville Function defining y'' for Lu=f.

    This form is used because it is expected for Scipy's solve_ivp method.

    Keyword arguments:
    x -- independent variable
    y -- dependent variable
    p -- p(x) parameter
    p_x -- derivative of p_x wrt x
    q -- q(x) parameter
    f -- forcing function f(x)
    alpha -- nonlinear parameter
    nonlinear_exp -- exponent of nonlinear term
    """
    y_x = y[1]
    y_xx = -1*(p_x/p)*y[1] + (q/p)*y[0] + (q/p)*alpha*y[0]**nonlinear_exp - f/p
    return [y_x, y_xx]


def euler_bernoulli_beam(x, y, EI, f):
    """Euler-Bernoulli Beam Theory defining y'''' for Lu=f.

    This form is used because it is expected for Scipy's solve_ivp method.

    Keyword arguments:
    x -- independent variable
    y -- dependent variable
    EI -- EI(x) parameter
    f -- forcing function f(x)
    """
    return [y[1], y[2], y[3], -1*f/EI]


def piecewise_p(x, val_a=25, val_b=300, val_c=100):
    """Define piecewise p(x) function.

    The p(x) function has three values at different spatial positions.

    Keyword arguments:
    x -- spatial position to evaluate piecewise p(x)
    val_a -- first value, for x>8 and 2<x<4
    val_b -- second value, for 6<x<8
    val_c -- third value, for 4<x<6 and 0<x<2

    Returns:
    p -- the value of p at a given position x
    """
    if x > 8:
        p = val_a
    elif x > 6:
        p = val_b
    elif x > 4:
        p = val_c
    elif x > 2:
        p = val_a
    else:
        p = val_c

    return p


def get_forcings(i_set: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                 j_set: List[float] = [1, 3, 5, 7, 10],
                 k_set: List[float] = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]):
    """Generate a collection of sine and cosine forcing functions.

    For each i, j, k in i_set, j_set, k_set, add the functions i*cos(j*x)+k and
    i*sin(j*x)+k to the list of forcing functions.

    Keyword arguments:
    i_set, j_set, k_set -- sets of numbers for computing forcing functions

    Returns:
    forcings -- a list of forcing functions
    """
    forcings = []
    for i in i_set:
        for j in j_set:
            for k in k_set:
                forcings.append(lambda x, i=i, j=j, k=k: i*np.cos(j*x)+k)
                forcings.append(lambda x, i=i, j=j, k=k: i*np.sin(j*x)+k)
    return forcings
