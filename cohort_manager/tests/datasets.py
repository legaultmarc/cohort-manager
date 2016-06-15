"""
Module of datasets for use in testing.
"""


import numpy as np


# Dataset of 10 samples.
types = {
    "continuous": [None, "-9", "1", "2.3", "5.6", None, "9.13", "8", "0.1",
                   "-2"],
    "discrete": [None, "1", None, "0", "1", "0", None, "1", "0", "0"],
    "integer": ["0", "31", None, "1", "12", "41", "65", None, "78", "-3"],
    "positiveinteger": ["2", "31", None, "1", "12", "41", "65", None, "78",
                        "3"],
    "negativeinteger": ["-2", "-3", None, "-1", "-12", "-1", "-6", None, "0",
                        "0"],
    "year": ["2013", None, "1998", "2001", "1999", "1974", "2021", "1978",
             None, "1988"],
    "factor": ["3", "2", "1", "1", "2", None, "3", "1", "2", None],
    "date": ["2016-01-23", "1998-12-31", "1752-06-01", "1874-02-23",
             "1999-07-21", "1963-11-16", "1924-02-14", "1984-09-19",
             "1955-01-01", "2260-05-19"],
    "pastdate": ["2015-01-23", "1998-12-31", "1752-06-01", "1874-02-23",
                 "1999-07-21", "1963-11-16", "1924-02-14", "1984-09-19",
                 "1955-01-01", "1964-05-19"],
}

n = np.nan
hierarchy = {
    "grandparent_discrete": [1, 0, 0, n, 1, 0, 1, 0, 1, 1],
    "parent_discrete": [0, n, n, n, 1, n, 0, n, 1, 1],
    "child_continuous": [n, n, n, n, 0.12, n, n, n, n, n],
    "child_discrete": [n, n, n, n, 0, n, n, n, 1, n],
    "child_date": [n, n, n, n, "2016-01-26", n, n, n, "2014-03-31", n],
    "child_continuous": [n, n, n, n, 0.12, n, n, n, n, n],
}


def fill_hierarchy(manager):
    manager.set_samples(list("abcdefghij"))
    manager.add_phenotypes([
        dict(name="grandparent_discrete", variable_type="discrete"),
        dict(name="parent_discrete", variable_type="discrete",
             parent="grandparent_discrete"),
        dict(name="child_date", variable_type="date",
             parent="parent_discrete"),
        dict(name="child_discrete", variable_type="discrete",
             parent="parent_discrete"),
        dict(name="child_continuous", variable_type="continuous",
             parent="parent_discrete"),
    ])

    for k, values in hierarchy.items():
        manager.add_data(k, values)
