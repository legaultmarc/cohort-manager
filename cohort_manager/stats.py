"""
Statistical utilities for the cohort manager.
"""

import numpy as np


def median_absolute_deviation(values, q=None, return_median=False):
    """Compute the median absolute deviation.

    Reference:
    C Leys, C Ley, O Klein, P Bernard, L Licata. Detecting outliers: Do not use
    standard deviation around the mean, use absolute deviation around the
    median. (2013) Jour. of Exper. Soc. Psy.

    """
    if q is None:
        q = 1.482602218505602  # 1.0 / scipy.stats.norm.ppf(0.75)
    median = np.median(values)
    mad = q * np.median(np.abs(values - median))
    if return_median:
        return (median, mad)
    return mad
