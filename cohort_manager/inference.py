"""
Module to facilitate inference about:
    - Structure between phenotypes
    - Variable data types
"""

import collections

import numpy as np

from . import stats


def bool_similarity(v1, v2):
    """Similarity measure of two boolean vectors."""
    nan_sim = np.isnan(v1) == np.isnan(v2)
    return 2 * (np.sum(v1 == v2) + nan_sim) / (v1.shape[0] + v2.shape[0])


def estimate_num_distinct(v):
    """Estimate the number of distinct elements in vector."""
    v = v[~np.isnan(v)]
    if v.shape[0] <= 5000:
        return len(set(v))
    else:
        # len(np.unique) is ~3x faster than len(set(x))
        return len(np.unique(np.random.choice(v, 5000, replace=False)))


def find_overrepresented_outlier(v):
    """Finds overrepresented outliers.

    This returns the overrepresented outlier if present. It returns None if
    it doesn't find anything suspicious.

    These values could mean that a symbol for missing values (e.g. -9) was
    used.
    """
    median, mad = stats.median_absolute_deviation(v, return_median=True)

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


def infer_variable_type(li, max_size=5000):
    """Infer a suitable data type from a list of strings."""

    # We limit the size to 'max_size' for the inference.
    n = len(li)
    if n > max_size:
        n = max_size
        li = np.random.choice(li, max_size, replace=False)

    counts = {
        "numeric": 0,
        "text": 0
    }

    # Check the number of distinct values.
    n_distinct = estimate_num_distinct(np.array(li))
    if n_distinct == 2:
        return "discrete"

    for i in li:
        try:
            int(i)
            counts["numeric"] += 1
            continue
        except ValueError:
            pass

        try:
            float(i)
            counts["numeric"] += 1
            continue
        except ValueError:
            pass

        counts["test"] += 1

    if counts["numeric"] / n > 0.85:
        return "continuous"

    # We assome it's text data. We only say factor if there is a reasonable
    # number of unique occurences.
    if n_distinct <= 5:
        return "factor"

    return None
