"""
Statistical utilities for the cohort manager.
"""

import collections

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


def find_overrepresented_outlier(v):
    """Finds overrepresented outliers.

    This returns the overrepresented outlier if present. It returns None if
    it doesn't find anything suspicious.

    These values could mean that a symbol for missing values (e.g. -9) was
    used.
    """
    median, mad = median_absolute_deviation(v, return_median=True)

    outliers = v[(v < median - 3 * mad) | (v > median + 3 * mad)]

    n = outliers.shape[0]
    # We tolerate a rate of outliers of less than 1%.
    if n <= 0.01 * v.shape[0]:
        return None

    counter = collections.Counter(outliers)
    most_common_outlier, count = counter.most_common(1)[0]

    # If the most common outlier is more than half the cases, we return it.
    if count >= n / 2:
        return most_common_outlier
