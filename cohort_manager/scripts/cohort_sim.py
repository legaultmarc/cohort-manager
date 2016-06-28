"""
Utility to simulate phenotype data.
"""

import datetime
import argparse
import random
import string

import scipy.stats
import numpy as np

import cohort_manager.types as types
from cohort_manager.core import CohortManager


def main(args):
    manager = CohortManager(args.cohort)

    # Create the samples.
    manager.set_samples(["sample_{}".format(i + 1) for i in range(args.n)])

    # Generate the variables.
    for i in range(args.n_variables):
        _simulate_variable(manager)


def _simulate_variable(manager, _type=None):
    # Generate a random name.
    name = "".join(
        [random.choice(string.ascii_lowercase) for i in range(5)]
    ).capitalize()

    # Choose a random type.
    if _type is None:
        _type = random.choice([types.Discrete, types.Continuous, types.Date,
                               types.Year])

    # Insert in database.
    manager.add_phenotype(name=name, variable_type=_type.__name__)

    # Simulate data.
    if _type is types.Discrete:
        data = _simulate_discrete_vector(manager.n)
    elif _type is types.Continuous:
        data = _simulate_continuous_vector(manager.n)
    elif _type is types.Date:
        data = _simulate_date_vector(manager.n)
    elif _type is types.Year:
        data = _simulate_year_vector(manager.n)
    else:
        raise NotImplementedError(
            "Simulation routine for '{}' data type is not implemented."
            "".format(_type)
        )

    # Add the simulated data.
    manager.add_data(name, data)


def _missing_mask(n):
    # Simulate a missing rate.
    missing_rate = np.random.uniform(0, 0.07)
    return scipy.stats.bernoulli.rvs(
        missing_rate, size=n
    ).astype(bool)


def _simulate_discrete_vector(n):
    # Sample a prevalence.
    p = scipy.stats.beta.rvs(2, 5)

    data = scipy.stats.bernoulli.rvs(p, size=n).astype(float)

    data[_missing_mask(n)] = np.nan
    return data


def _simulate_continuous_vector(n):
    # Sample a mu.
    mu = np.random.uniform(0, 200)

    # Sample a sigma.
    sigma = np.random.uniform(0.01 * mu, 2 * mu)

    data = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=n)

    # Add error.
    residual_variance = np.random.uniform(0.01 * sigma, 0.6 * sigma)
    data += scipy.stats.norm.rvs(loc=0, scale=residual_variance, size=n)

    # Add missing.
    data[_missing_mask(n)] = np.nan
    return data


def _simulate_date_vector(n):
    years = scipy.stats.norm.rvs(loc=1980, scale=30, size=n).astype(int)
    months = np.random.randint(1, 13, size=n).astype(int)
    days = np.random.randint(1, 28, size=n).astype(int)  # TODO Do better...

    missing_rate = np.random.uniform(0, 0.07)

    dates = []
    for i in range(n):
        if random.random() < missing_rate:
            dates.append(np.nan)
        else:
            date = datetime.date(year=years[i], month=months[i], day=days[i])
            dates.append(date.strftime("%Y-%m-%d"))

    return dates


def _simulate_year_vector(n):
    years = np.round(scipy.stats.norm.rvs(loc=1980, scale=30, size=n))
    years[_missing_mask(n)] = np.nan
    return years


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        help="Name of the cohort containing the simulated data.",
        default="_SimulatedCohort"
    )

    parser.add_argument(
        "-n",
        help="The number of samples (default: %(default)s).",
        default=20000
    )

    parser.add_argument(
        "--filename", "-o",
        help="The output filename.",
        default="simulated_phenotypes.csv"
    )

    parser.add_argument(
        "--n-variables", "-m",
        help="The number of variables (default: %(default)s).",
        default=1000
    )

    args = parser.parse_args()

    main(args)
