"""Simple function for generating randomized initial conditions."""

# Third party imports
import numpy as np


def generate_random_ics(exp_range: int = 5,
                        repeats: int = 3,
                        start_from: int = 0,
                        eqn_order: int = 2):
    """
    Generate randomized initial conditions.

    Pick a random number from an normal distribution, multiply by
    a random number between 0 and 1, then multiply by a power of 10.

    Keyword Arguments:
    exp_range -- range(exp_range) is the set of integers used as exponents for
    10 to produce initial conditions of different magnitudes.
    repeats -- the number of times to repeat a specific power of 10
    start_from -- an integer added to the range(exp_range) to shift the
    magnitudes of the solutions.
    eqn_order -- the differential order of the equation to be solved
    with these initial conditions. e.g. Euler-Bernoulli beam theory is
    fourth order, so eqn_order=4 is used.

    Returns:
    ic_list -- a collection of randomized initial conditions.
    """
    ic_list = []
    for i in range(exp_range):
        for j in range(repeats):
            ics = []
            for k in range(eqn_order):
                ic_val = np.random.randn() * np.random.rand()
                ic_val = ic_val * 10**(i+start_from)
                ics.append(abs(ic_val))
            ic_list.append(ics)
    return ic_list
