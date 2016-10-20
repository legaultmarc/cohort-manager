"""
Module to facilitate inference about:
    - Structure between phenotypes
    - Variable data types
"""

import collections
import logging
import datetime
import re

import numpy as np
import scipy.cluster.hierarchy
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.special import digamma

from . import core
from . import types

plt.style.use("ggplot")
logger = logging.getLogger(__name__)


def infer_type(li, max_size=5000, known_missings=None):
    """Infer a suitable data type from a list of strings.

    :param li: A list of values.
    :type li: list (str)

    :param max_size: The maximum number of values to consider for inference.
    :type max_size: int (default 5000)

    :param known_missings: Strings that indicate missing values.
    :type known_missings: dict (default {"", "NA"})

    :returns: The inferred type for the list of values.
    :rtype: str

    Possible types are:

        - discrete
        - year
        - unique_key
        - continuous
        - text

    .. note::

        This function uses the assumption that the dataset is somewhat large.

    """
    # Remove known missing values.
    if known_missings is None:
        known_missings = {"", "NA"}
    else:
        known_missings = set(known_missings)

    li = [i for i in li if i not in known_missings]

    # We limit the size to 'max_size' for the inference.
    n = len(li)
    if n > max_size:
        n = max_size
        li = np.random.choice(li, max_size, replace=False)

    # We count both the distinct values and the occurences of every type.
    value_counts = collections.Counter(li)

    type_counts = collections.defaultdict(int)

    for value, count in value_counts.items():
        type_counts[infer_primitive_type(value)] += count

    if not type_counts:
        return None

    # This is not informative.
    if "null" in type_counts:
        del type_counts["null"]

    # Sort the types.
    type_counts = sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
    )

    _discrete = (
        type_counts[0][0] in ("positive_integer", "zero") and
        len(value_counts) <= 2
    )
    if _discrete:
        return "discrete"

    if type_counts[0][0] == "past_year":
        return "year"

    if type_counts[0][0] == "date":
        return "date"

    # Unique values suggest that this is the sample id.
    if len(value_counts) == n and type_counts[0][0] != "real":
        return "unique_key"

    # There is at least some heterogeneity.
    if len(value_counts) > 5:
        numeric_types = {"positive_integer", "negative_integer", "real",
                         "zero"}

        if not set([i[0] for i in type_counts]) - numeric_types:
            return "continuous"
        else:
            return "text"

    return None


def infer_primitive_type(value):
    """Infer the primitive type of the given string.

    :param value: The value of unknown type.
    :type value: str

    :returns: The inferred type (see below).
    :rtype: str

    The possible inferred types are:

        - null (empty string)
        - past_year (years between 1600 and the current year)
        - positive_integer
        - negative_integer
        - zero
        - real
        - date
        - string

    This works by testing a list of predicate functions that return True if
    they match the value and False otherwise. The first match in the above
    list is returned. This means that if the "past_year" predicate matches,
    the other types won't be tested.

    """
    types = collections.OrderedDict()

    # The first predicate matches the empty string and corresponds to the
    # "null" inferred type.
    types[lambda x: x == ""] = "null"

    int_regex = re.compile(r"^-?[0-9]+$")

    def _f(x):
        try:
            x = int(x)
            return 1600 <= x <= datetime.datetime.now().year
        except Exception:
            return False

    types[_f] = "past_year"

    def _f(x):
        if re.match(int_regex, x.strip()):
            return int(x) > 0
        return False

    types[_f] = "positive_integer"

    def _f(x):
        if re.match(int_regex, x.strip()):
            return int(x) < 0
        return False

    types[_f] = "negative_integer"

    def _f(x):
        try:
            v = int(x)
            return v == 0
        except Exception:
            return False

    types[_f] = "zero"

    def _f(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    types[_f] = "real"

    def _f(x):
        try:
            datetime.datetime.strptime(x, "%Y-%m-%d")
            return True
        except Exception:
            return False

    types[_f] = "date"

    for predicate, type_name in types.items():
        if predicate(value):
            return type_name

    return "string"


def cast_type(values, data_type):
    """Cast a list of values into a numpy array of the given type.

    This function will return the value mapping if the type is 'discrete'.

    """
    if data_type is None:
        return None, np.array(values)

    if data_type.subtype_of(types.Discrete):
        return _encode_discrete(values)

    if data_type.subtype_of(types.Continuous):
        out = []
        for i in values:
            try:
                out.append(float(i))
            except Exception:
                out.append(np.nan)
        return None, np.array(out, dtype=np.float)

    # encode and check both understand strings for the date type.
    if data_type.subtype_of(types.Date):
        return None, values

    raise ValueError(
        "Automatic type cast to '{}' is not yet supported.".format(
            data_type.__name__
        )
    )


def _encode_discrete(values):
    """Encode a list of string representing a discrete variable.

    Returns a dict representing inferred value mappings and a numpy array of
    values.

    """
    def _cast(v):
        try:
            return int(v)
        except Exception:
            return np.nan

    values = np.array([_cast(i) for i in values])

    counter = collections.Counter(values[~np.isnan(values)])
    symbols = set(counter.keys())

    # If symbols are 0, 1, then it's not ambiguous. We just cast.
    if symbols == {0, 1}:
        return {0: 0, 1: 1}, values

    # If it's all NaNs and symbols, we assume that affected individuals are
    # the ones with a value.
    if len(symbols) == 1:
        affected = symbols.pop()
        code = {1: affected}
        values[values == affected] = 1
        return code, values

    # If there are two symbols that are not 0 and 1s, we take the most frequent
    # as denoting unaffected individuals.
    if len(symbols) == 2:
        unaffected = counter.most_common()[0][0]
        affected = counter.most_common()[1][0]
        code = {1.0: affected, 0.0: unaffected}
        return code, core.vector_map(
            values, [(unaffected, 0.0), (affected, 1.0)]
        )

    # Unable to encode as a discrete variable.
    raise ValueError("Unable to encode values as a discrete variable.")


def build_mi_matrix(dataset, inferred_types):
    """Build the pairwise mutual information matrix.

    The dataset is a list of (col1, col2, mutual information).

    """
    logger.info("Building the variable relationship matrix. This can take a "
                "long time.")

    names = list(dataset.keys())
    weights = []
    for i, col1 in enumerate(names):
        for j, col2 in enumerate(names):
            if i < j:
                type1 = inferred_types[col1]
                type2 = inferred_types[col2]

                w = mutual_information(v1=dataset[col1], type1=type1,
                                       v2=dataset[col2], type2=type2)

                if w is None:
                    w = 0

                weights.append(w)

    return names, np.array(weights)


def hierarchical_clustering(mat, examples, inferred_types, codes):
    names, weights = mat
    weights[weights == 0] = 0.01
    distances = 1 / weights

    linkage = scipy.cluster.hierarchy.linkage(distances)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True,
                             gridspec_kw=dict(width_ratios=(0.35, 0.65)))

    # Plot the tree.
    tree_info = scipy.cluster.hierarchy.dendrogram(linkage, ax=axes[0],
                                                   orientation="left")

    # Plot the dataset.
    m = len(names)  # number of variables.
    n = next(iter(examples.values())).shape[0]  # number of samples.
    rows = tree_info["leaves"]
    labels = ["{} - {}".format(s, names[s]) for s in rows]
    axes[0].set_yticklabels(labels)
    axes[0].tick_params(labelsize=9)

    # Create the grid.
    yticks = axes[0].get_yticks()
    x, y = np.meshgrid(
        np.arange(n + 1),
        np.linspace(yticks[0], yticks[-1], m + 1)
    )
    y += 1

    z = None
    for col in rows:
        col = names[col]

        try:
            data_type = types.type_str(inferred_types[col])
        except Exception:
            data_type = None

        v = examples[col].copy()
        try:
            v[np.isnan(v)] = -1
        except TypeError:
            v = np.full_like(v, -1, dtype=float)

        # Only discrete variables are plotted.
        if data_type and data_type.subtype_of(types.Discrete):
            # Remap wrt. to code.
            v[v == 0] = 0.5
            v[v == -1] = 0
        else:
            v = np.full_like(v, -1, dtype=float)

        # z is the full matrix we're plotting.
        if z is None:
            z = v
        else:
            z = np.vstack((z, v))

    cmap = matplotlib.colors.ListedColormap(
        ["#FFFDEE", "#43728C", "#E0473D"],
        name="Yabadabadoo",
    )
    bounds = [0, 0.4, 0.6, 1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    axes[1].pcolormesh(x, y, z, cmap=cmap, norm=norm)

    plt.tight_layout()
    logger.info("Generated the 'variable_plot.png' file illustrating the "
                "hierarchical clustering of the phenotype data.")

    plt.savefig("variable_plot.png", dpi=600)

    threshold = np.sort(np.unique(linkage[:, 2]))[-2]
    return scipy.cluster.hierarchy.fcluster(linkage, threshold, "distance")


def mutual_information(v1, type1, v2, type2, normalized=True):
    """Dispatch to the proper mutual information estimator.

    :param v1: Samples from the first distribution.
    :type v1: numpy.ndarray

    :param type1: The type of the first variable.
    :type type1: str

    :param v2: Samples from the second distribution.
    :type v2: numpy.ndarray

    :param type2: The type of the second variable.
    :type type2: str

    Recognized types are:
        - continuous
        - year (interpreted as continuous)
        - discrete

    """
    _continuous = ("continuous", "year")

    mi = None
    if type1 == "discrete" and type2 == "discrete":
        mi = _discrete_discrete_mi(v1, v2, normalized=normalized)

    if type1 in _continuous and type2 == "discrete":
        mi = _discrete_continuous_mi(v2, v1, normalized=normalized)

    if type2 in _continuous and type1 == "discrete":
        mi = _discrete_continuous_mi(v1, v2, normalized=normalized)

    if type1 in _continuous and type2 in _continuous:
        mi = _continuous_continuous_mi(v1, v2, normalized=normalized)

    return mi


def _discrete_discrete_mi(v1, v2, ignore_na=False, normalized=True):
    """Compute the mutual information in bits between two vectors."""
    if not ignore_na:
        v1_nan = np.isnan(v1)
        v2_nan = np.isnan(v2)
        v1[v1_nan] = -1
        v2[v2_nan] = -1

    else:
        nans = np.isnan(v1) | np.isnan(v2)
        v1 = v1[~nans]
        v2 = v2[~nans]

    xdom = list(set(v1))
    ydom = list(set(v2))

    n = v1.shape[0]
    i = 0
    h_x = 0
    h_y = 0
    for x in xdom:
        p_x = np.sum((v1 == x)) / n
        h_x += p_x * np.log2(p_x)

        for y in ydom:
            p_y = np.sum((v2 == y)) / n
            if x == -1:  # Do this only for the first outer iteration.
                h_y += p_y * np.log2(p_y)

            p_xy = np.sum((v1 == x) & (v2 == y)) / n

            if p_xy > 0:
                i += p_xy * np.log2(p_xy / (p_x * p_y))

    i = max(0, i)

    ret = i

    if normalized:
        h_x = discrete_entropy(v1)
        h_y = discrete_entropy(v2)
        ret = i / (h_x + h_y)

    # Reassign the NAs to avoid side effects.
    if not ignore_na:
        try:
            v1[v1_nan] = np.nan
            v2[v2_nan] = np.nan
        except ValueError:
            pass

    return ret


def _discrete_continuous_mi(d, c, k=3, normalized=True):
    """Compute MI between a continuous and a discrete variable.

    :param d: The discrete vector.
    :param c: The continuous vector.
    :parma k: Number of samples to consider for the nearest neighbor (see the
              paper for a description).

    Implements method described in:

    Ross, BC. Mutual Information between Discrete and Continuous Data Sets
    (2014) PLoS ONE

    """
    # We recode missing values for the discrete variable.
    original_d = d
    d_nan = np.isnan(d)
    d[d_nan] = -1

    # We mask the missing values for the continuous variable.
    nans = np.isnan(c)
    d = d[~nans]
    c = c[~nans]

    # Sort wrt continuous variable to make faster distance computation.
    _sort_keys = np.argsort(c)
    d = d[_sort_keys]
    c = c[_sort_keys]

    n = d.shape[0]
    assert n == c.shape[0], "Vectors are of different shape."

    # n_xi is the number of points with the same value of the discrete
    # variable as the current x.
    # This dict is used for caching.
    n_xi = {}

    I = np.empty(n)

    for i in range(n):
        # Get the number of points with the same value as this point
        # (N_{x_i}).
        cur_n_xi = n_xi.get(d[i])
        if not cur_n_xi:
            cur_n_xi = np.sum(d == d[i])
            n_xi[d[i]] = cur_n_xi

        # In this subset of values, we take the k nearest neighbours and
        # compute d (the distance to the kth NN).
        # We then compute the number of points in the continuous dataset
        # that are closer than d ().
        dist = _find_distance(c, d, i, k)
        end_index = np.searchsorted(c, c[i] + dist, side="right")
        start_index = np.searchsorted(c, c[i] - dist, side="left")
        m_i = end_index - start_index - 1  # Remove the current point.

        I[i] = digamma(n) - digamma(cur_n_xi) + digamma(k) - digamma(m_i)

    mi = max(np.mean(I), 0)
    ret = mi

    if normalized:
        ret = mi / (discrete_entropy(d) + continuous_entropy(c))

    # Avoid side effects.
    try:
        original_d[d_nan] = np.nan
    except ValueError:
        pass

    return ret


def _continuous_continuous_mi(u, v, normalized=True):
    """Compute mutual information between two continuous vectors.

    This is done by using:

        I(X, Y) = H(X) + H(Y) - H(X, Y)

    The entropy is computed using the ``entropy`` method from this module.

    """
    nans = np.isnan(u) | np.isnan(v)
    u = u[~nans]
    v = v[~nans]
    uv = np.hstack((u, v)).reshape(u.shape[0], 2)
    entropy = continuous_entropy
    mi = max(entropy(u) + entropy(v) - entropy(uv), 0)

    if normalized:
        return mi / (continuous_entropy(u) + continuous_entropy(v))

    return mi


def discrete_entropy(v):
    """Compute the entropy of discrete variables.

    This function is only implemented for 1D.

    """
    if len(v.shape) > 1:
        raise NotImplementedError("Joint discrete entropy is not implemented.")
    n = v.shape[0]

    domain = np.unique(v)
    acc = 0
    for x in domain:
        p_x = np.sum(v == x) / n
        acc += p_x * np.log2(p_x)

    return -acc


def continuous_entropy(data, bounds=None):
    """Compute the entropy of the features of a dataset.

    :param data: A matrix of n_samples x n_features (or a 1d vector).
    :type data: numpy.ndarray

    This is inspired by the implementation from:
    https://github.com/msmbuilder/mdentropy

    .. note::
        This method uses a gaussian kernel density estimate of the
        distribution.

    Joint entropy is returned if there are multiple features.

    """
    # If 1D vector, convert to 2D.
    if len(data.shape) == 1:
        data = data.reshape(data.shape[0], 1)

    # Build a grid covering the whole domain of the dataset.
    if bounds is None:
        bounds = list(zip(np.min(data, axis=0), np.max(data, axis=0)))

    n_points = 50
    grid = np.meshgrid(*[np.linspace(a, b, n_points) for a, b in bounds])

    # Take the KDE of the dataset.
    kde = scipy.stats.gaussian_kde(data.T)

    # Apply the KDE to the mesh.
    # kde takes data in the form feature x sample, so we want the mesh to
    # be expressed as [[x1, x2, ...],
    #                  [y1, y2, ...], ...]
    densities = kde(np.vstack([np.ravel(i) for i in grid]))

    # Reshape to fit the grid.
    densities = densities.reshape(*grid[0].shape)
    # Set the densities of 0 to 1. They will not count anyway because log(1)=0.
    densities[densities == 0] = 1

    # Compute the entropy.
    dx = np.product([abs(b - a) / n_points for a, b in bounds])
    return -np.nansum(densities * np.log2(densities)) * dx


def _find_distance(c, d, i, k):
    """Find the distance of the kth nearest neighbour."""
    # Look k to the left and k to the right.
    distances = []

    x_i = d[i]
    y_i = c[i]

    # Get k valid distances to the left.
    cur = i - 1
    while len(distances) < k:
        if cur < 0:
            break
        if d[cur] == x_i:
            distances.append(abs(c[cur] - y_i))
        cur -= 1

    # Get k valid distances to the right.
    cur = i + 1
    while len(distances) < 2 * k:
        if cur == d.shape[0]:
            break
        if d[cur] == x_i:
            distances.append(abs(c[cur] - y_i))
        cur += 1

    try:
        return sorted(distances)[k - 1]
    except IndexError:
        return 0
