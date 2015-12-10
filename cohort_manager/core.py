"""
"""

from __future__ import division

import collections
import logging
import sqlite3
import os

import six
import h5py
import numpy as np
import pandas as pd

from .phenotype_tree import PHENOTYPE_COLUMNS, tree_from_database
from . import stats


logger = logging.getLogger(__name__)


class UnknownSamplesError(Exception):
    def __init__(self, value=None):
        self.value = value

    def __str__(self):
        if self.value is None:
            # Default message.
            return ("Sample order was not set. It needs to be known before "
                    " data is added to enforce integrity checks.")
        return self.value


class FrozenCohortError(Exception):
    def __str__(self):
        return ("Once the sample order has been set, it is impossible to "
                "change without rebuilding the database.")


class CohortDataError(Exception):
    def __init__(self, value=None):
        self.value = value

    def __str__(self):
        if self.value is None:
            return ("An invalid setting was detected in your cohort manager.")

        return self.value


class CohortManager(object):
    def __init__(self, name, path=None):
        self.name = name
        self.path = path if path else os.path.abspath(name)
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self._discover_install()
        self.tree = None  # This is built when the manager is commited.

    def _discover_install(self):
        """Check if there is a retrievable manager at location.

        If there is none, create the database.

        """
        self.db_path = os.path.join(self.path, "phenotypes.db")
        db_init = not os.path.isfile(self.db_path)

        self.data_path = os.path.join(self.path, "data.hdf5")
        data_init = not os.path.isfile(self.data_path)

        self.con = sqlite3.connect(self.db_path)
        self.cur = self.con.cursor()

        self.data = h5py.File(self.data_path)

        if db_init:
            self._create()

        if data_init:
            self._hdf5_init()

        if db_init and data_init:
            logger.info("Detecting cohorts... [NEW COHORT]")
        else:
            logger.info("Detecting cohorts... [RESTORE COHORT]")

        logger.info("Database file is: {}".format(self.db_path))
        logger.info("Data file is: {}".format(self.data_path))

    def _create(self):
        """SQL CREATE statements."""
        logger.info("Creating tables...")
        # Warning: If you change the order of columns, you should also change
        # the PHENOTYPE_COLUMNS constant in the phenotype_tree module.
        self.cur.execute(
            "CREATE TABLE phenotypes ("
            " name TEXT PRIMARY KEY,"
            " icd10 TEXT,"
            " parent TEXT,"
            " variable_type TEXT,"
            " crf_page INTEGER,"
            " question TEXT,"
            " code_name TEXT,"
            " FOREIGN KEY(parent) REFERENCES phenotypes(name)"
            ");"
        )

        self.cur.execute(
            "CREATE TABLE code ("
            " name TEXT,"
            " key INT,"
            " value TEXT,"
            " CONSTRAINT code_pk PRIMARY KEY (name, key, value)"
            ");"
        )

        self.cur.execute(
            "CREATE TABLE app_meta ("
            " key TEXT,"
            " value TEXT,"
            " CONSTRAINT app_meta_pk PRIMARY KEY (key, value)"
            ");"
        )

        self.con.commit()

    def _hdf5_init(self):
        """Create the relevant HDF5 datasets."""
        self.data.create_group("data")

    def _check_phenotype_fields(self, fields):
        extra = set(fields) - set(PHENOTYPE_COLUMNS)

        if extra:
            raise TypeError("Unexpected column(s) '{}'.".format(extra))

        if "name" not in fields:
            raise TypeError("Expected a 'name' column (primary key).")

    def __getitem__(self, key):
        self.cur.execute("SELECT * FROM app_meta WHERE key=?;", (key, ))
        element = self.cur.fetchone()
        try:
            return element[1]
        except Exception:
            return element

    def __setitem__(self, key, value):
        # TODO: Cast to proper type.
        if self[key]:
            self.cur.execute(
                "UPDATE app_meta SET value=? WHERE key=?",
                (value, key)
            )

        else:
            self.cur.execute(
                "INSERT INTO app_meta VALUES (?, ?)",
                (key, value)
            )

        self.con.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.close()

    def close(self):
        self.data.close()
        self.con.commit()
        self.con.close()

    # Public API.
    @property
    def n(self):
        if self["frozen"] == "yes":
            return self.get_samples().shape[0]
        return None

    def rebuild_tree(self):
        """Rebuild the phenotype tree."""
        self.tree = tree_from_database(self.cur)

    def commit(self):
        """Commit the database and rebuilt the phenotype tree."""
        self.con.commit()
        self.rebuild_tree()

    # Add information.
    def add_phenotype(self, **kwargs):
        """Insert a phenotype into the database.

        Known fields are:
            - name
            - icd10
            - parent
            - variable_type
            - crf_page
            - question
            - code_name

        .. todo::
            Check that the ``code_name`` is the the ``code`` table.

        """
        self._check_phenotype_fields(kwargs.keys())
        values = map(kwargs.get, PHENOTYPE_COLUMNS)

        self.cur.execute(
            "INSERT INTO phenotypes VALUES (?, ?, ?, ?, ?, ?, ?)",
            tuple(values)
        )

    def add_phenotypes(self, li):
        """Batch insert phenotypes into the database."""
        for element in li:
            self.add_phenotype(**element)

    def add_code(self, name, key, value):
        """Insert a code (INTEGER -> phenotype)."""
        self.cur.execute(
            "INSERT INTO code VALUES (?, ?, ?)",
            (str(name), int(key), str(value))
        )

    def add_codes(self, li):
        """Batch insert codes into the database."""
        for element in li:
            self.add_code(*element)

    def set_samples(self, samples):
        """Set the samples.

        Once this is set, it can't be changed without rebuilding the database.

        """
        if samples:
            assert isinstance(samples[0], six.string_types), (
                "Sample IDs need to be strings."
            )

        if self["frozen"] == "yes":
            raise FrozenCohortError()

        self.data.create_dataset("samples", data=samples)
        self["frozen"] = "yes"

    def add_data(self, phenotype, values):
        """Add a data vector to the cohort."""
        # Check if phenotype is in database (metadata needs to be there before
        # the user binds data).
        self.cur.execute(
            "SELECT name, variable_type, code_name "
            "FROM phenotypes WHERE name=?",
            (phenotype, )
        )
        tu = self.cur.fetchone()
        if tu is None:
            raise ValueError(
                "Could not find metadata for '{}'. Add the phenotype before "
                "binding data.".format(phenotype)
            )
        assert tu[0] == phenotype
        variable_type = tu[1]
        code_name = tu[2]

        if self["frozen"] != "yes":
            raise UnknownSamplesError()

        n = self.n
        if len(values) != n:
            raise ValueError(
                "Expected {} values, got {}.".format(n, len(values))
            )

        if phenotype in self.data["data"].keys():
            raise ValueError("Data for '{}' is already "
                             "in the database.".format(phenotype))

        # Type check.
        if variable_type == "discrete":
            # We are expecting only 0, 1 and NA.
            self._check_data_discrete(values, phenotype)

        elif variable_type == "factor":
            # Make sure the matadata code is consistent with the observed
            # values.
            self._check_data_factor(values, phenotype, code_name)

        elif variable_type == "continuous":
            # If there is less than 5 distinct values, warn the user and
            # suggest using a different variable type.
            self._check_data_continuous(values, phenotype)

        else:
            raise ValueError("Unknown variable type '{}'. Use 'discrete', "
                             "'factor' or 'continuous'.".format(variable_type))

        self.data["data"].create_dataset(phenotype, data=values)

    def _check_data_discrete(self, values, phenotype):
        """Check that the data vector is consistent with a discrete variable.

        This is done by making sure that only 0, 1 and NAs are in the vector.

        """
        extra = set(values[~np.isnan(values)]) - {0, 1}
        if len(extra) != 0:
            extra = ", ".join([str(i) for i in list(extra)])
            raise ValueError(
                "Authorized values for discrete variables are 0, 1 and np.nan."
                "\nUnexpected values were observed for '{}' ({})."
                "".format(phenotype, extra)
            )

    def _check_data_factor(self, values, phenotype, code_name):
        """Check that the data vector is consistent with a factor variable.

        This is done by looking at the code from the database and making sure
        that all the observed integer codes are defined.

        """
        # Get the code.
        self.cur.execute(
            "SELECT key, value FROM code WHERE name=?",
            (code_name, )
        )
        mapping = self.cur.fetchall()
        if not mapping:
            raise CohortDataError("Could not find the code for factor "
                                  "variable '{}'. The code name is "
                                  "'{}'.".format(phenotype, code_name))

        # Check if the set of observed values is consistent with the code.
        observed = set(values[~np.isnan(values)])
        expected = set([i[0] for i in mapping])
        extra = observed - expected
        if extra:
            raise CohortDataError("Unknown encoding value(s) ({}) for factor "
                                  "variables '{}'.".format(extra, phenotype))

    def _check_data_continuous(self, values, phenotype):
        """Check that the data vector is continuous.

        This is very hard to check.

        After taking the outlier data points (arbitrarily defined as):

            |x| > 3 * median absolute deviation + median

        We take the most common outlier if it consists of at least 50% of the
        identified data points. If such a value exist, we suggest to the
        user that it might be a weird value arising from bad missing variable
        coding.

        Also, if there is less than 5 distinct values in either the full vector
        or a random sample of 5000 elements, a warning will be displayed
        prompting the user to verify the data type.

        """
        values = values[~np.isnan(values)]

        # Factor check.
        # Sample up to 5000 samples to see the number of distinct values.
        if values.shape[0] > 5000:
            sample_set = set(values)
        else:
            sample_set = set(np.random.choice(values, 5000))

        if len(sample_set) < 5:
            logger.warning("The phenotype '{}' is marked as continuous, but "
                           "it has a lot of redundancy. Perhaps it should be "
                           "modeled as a factor or another variable type."
                           "".format(phenotype))

        # Outlier check.
        median, mad = stats.median_absolute_deviation(
            values, return_median=True
        )
        outliers = values[
            (values < median - 3 * mad) | (values > median + 3 * mad)
        ]

        n_outliers = outliers.shape[0]
        if n_outliers <= 5:
            return

        counter = collections.Counter(outliers)
        most_common_outlier, count = counter.most_common(1)[0]

        if count >= n_outliers / 2:
            logger.warning("The value '{}' is commonly found in the tails "
                           "of the distribution for '{}'. This could be "
                           "because of bad coding of missing values."
                           "".format(most_common_outlier, phenotype))

    # Get information.
    def get_samples(self):
        """Get the ordered samples."""
        try:
            return self.data["samples"]
        except KeyError:
            return None

    def get_phenotype(self, phenotype):
        """Get information on the phenotype."""
        self.cur.execute(
            "SELECT * FROM phenotypes WHERE name=?;",
            (phenotype, )
        )
        out = self.cur.fetchone()
        out = dict(zip(PHENOTYPE_COLUMNS, out))

        return out

    def get_phenotypes_list(self):
        """Get a list of available phenotypes from the db."""
        self.cur.execute("SELECT name FROM phenotypes;")
        li = self.cur.fetchall()
        if li:
            li = [tu[0] for tu in li]
        return li

    def get_code(self, name):
        """Get the integer mappings for a given code."""
        self.cur.execute("SELECT key, value FROM code WHERE name=?", (name, ))
        return self.cur.fetchall()

    def get_data(self, phenotype, numpy=False):
        """Get a phenotype vector.

        By default, this will be the HDF5 dataset because it's the most
        efficient representation.

        If you need the numpy functionalities, use `numpy=True`.

        """
        try:
            data = self.data["data/" + str(phenotype)]
        except KeyError:
            raise KeyError("No data for '{}'.".format(phenotype))

        if not numpy:
            return data

        data = np.array(data)
        # Get metadata.
        meta = self.get_phenotype(phenotype)
        if meta["variable_type"] in ("continuous", "discrete"):
            return data

        return self._represent_factor_data(data, meta)

    def _represent_factor_data(self, data, meta):
        # Get the code.
        code = self.get_code(meta["code_name"])
        # Check to see if the code is eligible to directly create a pandas
        # categorical variable.
        keys = [i[0] for i in code]
        if set(keys) == set(range(len(keys))):
            # Code is valid (maps to integers 0, 1, ..., n)
            # Recode missings for pandas convention.
            assert -1 not in keys
            data[np.isnan(data)] = -1
            return pd.Categorical.from_codes(
                data,
                [i[1] for i in sorted(code, key=lambda x: x[0])]
            )

        # If not, we fallback to the default constructor.
        # This is less efficient...
        code = dict(code)
        return pd.Series(
            [code[i] if not np.isnan(i) else np.nan for i in data],
            dtype="category"
        )

    def get_code_names(self):
        """Get a set of code names."""
        self.cur.execute(
            "SELECT DISTINCT name FROM code;"
        )
        return set([i[0] for i in self.cur.fetchall()])

    def validate(self, mode="raise"):
        """Run a batch of data validation checks.

        This will make sure that:

        - Samples marked as unaffected for a parent are marked as unaffected
          for all children if both variable types are discrete.
        - All the phenotypes in the database have associated data.

        A mode can be provided that will determine if a CohortDataError is
        raised (mode="raised", default) or if a warning is displayed
        (mode="warn").

        """
        if mode not in ("raise", "warn"):
            raise ValueError("Invalid mode type for check ('{}'). Valid modes "
                             "are 'warn' and 'raise'.".format(mode))

        if mode == "warn":
            _print = logger.warning
        else:
            def _print(s):
                raise CohortDataError(s)

        self.check_phenotypes_have_data(_print)
        self.hierarchical_reclassify()

    def check_phenotypes_have_data(self, printer):
        """Check that all the phenotypes in the database have data entries."""
        logger.info("Making sure that data is available for all phenotypes "
                    "in the database.")
        missing = set()
        available = set(self.data["data"].keys())
        self.cur.execute("SELECT name FROM phenotypes")
        for tu in self.cur:
            name = tu[0]
            if name not in available:
                missing.append(name)

        if missing:
            printer("Missing data for phenotypes '{}'.".format(missing))
            return False
        logger.debug("Phenotype in the database are consistent with the "
                     "binary data store.")
        return True

    def hierarchical_reclassify(self):
        """Reclassify controls according to the PhenotypeTree.

        Let u, v be discrete phenotypes, where u is the parent of v.
        if patient i in unaffected for u, he should be unaffected for v.
        This function adjusts the data container to reflect this constraint.

        """
        logger.info("Reclassifying case/control data with respect to the "
                    "hierarchical structure in the phenotype description.")
        if self.tree is None:
            raise Exception("Commit the manager to generate the hierarchical "
                            "phenotype representation.")

        # Use a depth first traversal to reclassify controls.
        for node in self.tree.depth_first_traversal():
            if node.parent is None:
                continue

            # Check if child and parent are discrete.
            if node.data.variable_type != "discrete":
                continue

            if node.parent.data.variable_type != "discrete":
                continue

            data = np.array(self.data["data/" + node.data.name])
            missing = np.isnan(data)

            parent_data = np.array(self.data["data/" + node.data.parent])
            parent_control = parent_data == 0

            # Reclassification.
            data[missing & parent_control] = 0

            # Write to HDF5
            self.data["data/" + node.data.name][...] = data
