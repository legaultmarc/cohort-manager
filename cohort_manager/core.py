"""
"""

from __future__ import division

import collections
import itertools
import logging
import sqlite3
import atexit
import uuid
import os

import six
import h5py
import numpy as np
import pandas as pd
from six.moves import range

from .phenotype_tree import PHENOTYPE_COLUMNS, tree_from_database
from . import inference
from .drugs.chembl import ChEMBL


logger = logging.getLogger(__name__)
VARIABLE_TYPES = {"discrete", "continuous", "factor"}


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
        return ("Once the sample order has been set, it is fixed and can't "
                "be changed.")


class CohortDataError(Exception):
    def __init__(self, value=None):
        self.value = value

    def __str__(self):
        if self.value is None:
            return ("An invalid setting was detected in your cohort manager.")

        return self.value


class Permutation(object):
    """Utility class to define an arbitrary sample order for data access."""
    def __init__(self, manager, samples_list, allow_subset=False):
        self._manager = manager
        self.samples = samples_list
        self.allow_subset = allow_subset

        self.index_map = self._build_index_map()

    def get_data(self, phenotype):
        data = self._manager.get_data(phenotype, numpy=True)
        return data[self.index_map]

    def _build_index_map(self):
        set_samples = set(self.samples)
        set_manager_samples = set(self._manager.get_samples())

        extra = set_samples - set_manager_samples
        if extra:
            raise ValueError(
                "Phenotypes are unavailable for {} of the requested samples "
                "(in the new order sequence).".format(len(extra))
            )

        missing = set_manager_samples - set_samples
        message = ("Phenotypes for {} samples were discarded in the "
                   "permutation.".format(len(missing)))
        if missing and not self.allow_subset:
            raise ValueError(message)
        elif missing:
            logger.info(message)

        manager_indices = {
            sample: i for i, sample in enumerate(self._manager.get_samples())
        }
        return np.array([manager_indices[i] for i in self.samples])


class CohortManager(object):
    def __init__(self, name, path=None):
        self.name = name
        if path:
            self.path = os.path.join(path, name)
        else:
            self.path = os.path.abspath(name)

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # We use caching to save expensive decodes.
        self._cache = {"samples": None}

        self._discover_install()
        self.tree = None  # This is built when the manager is commited.

        atexit.register(self._hdf5_close)

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
            " variable_type TEXT NOT NULL,"
            " crf_page INTEGER,"
            " description TEXT,"
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

        self.cur.execute(
            "CREATE TABLE drug_users ("
            " drug_id TEXT,"
            " sample_id TEXT,"
            " CONSTRAINT drug_users_pk PRIMARY KEY (drug_id, sample_id)"
            ");"
        )

        self.cur.execute(
            "CREATE TABLE dummy_phenotypes ("
            " name TEXT PRIMARY KEY,"
            " FOREIGN KEY(name) REFERENCES phenotypes(name)"
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

        if "variable_type" not in fields:
            raise TypeError("Expected a 'variable_type' column.")

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

    def _hdf5_close(self):
        try:
            self.data.close()
        except Exception:
            pass

    def close(self):
        self._hdf5_close()
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
            - description
            - code_name

        .. todo::
            Check that the ``code_name`` is in the ``code`` table.

        """
        self._check_phenotype_fields(kwargs.keys())
        values = map(kwargs.get, PHENOTYPE_COLUMNS)

        # Check that the type is valid.
        if kwargs["variable_type"] not in VARIABLE_TYPES:
            raise TypeError("Unknown variable type '{}'.".format(
                kwargs["variable_type"]
            ))

        # Factor variables need a code_name.
        if kwargs["variable_type"] == "factor":
            if kwargs.get("code_name") is None:
                raise TypeError("Factor variables need a 'code_name'.")

        self.cur.execute(
            "INSERT INTO phenotypes VALUES (?, ?, ?, ?, ?, ?, ?)",
            tuple(values)
        )

    def add_dummy_phenotype(self, name):
        """Insert a dummy phenotype into the database.

        .. note::
            The dummy phenotype is added to the "normal" phenotype table, but
            an entry is added to the dummy_phenotype table.

        """
        # Adding the dummy phenotype to the normal phenotype table
        self.cur.execute(
            "INSERT INTO phenotypes (name, variable_type) VALUES (?, ?)",
            (name, "dummy"),
        )

        # Adding the dummy phenotype to the dummy phenotype table
        self.cur.execute(
            "INSERT INTO dummy_phenotypes (name) VALUES (?)",
            (name, ),
        )

    def register_drug_user(self, drug_id, sample):
        """Register that 'sample' is a user of drug 'drug_id'"""
        if sample not in self.get_samples():
            raise ValueError("Sample '{}' not in the manager.".format(sample))
        self.cur.execute(
            "INSERT INTO drug_users VALUES (?, ?)", (str(drug_id), sample)
        )

    def update_phenotype(self, name, **kwargs):
        """Update an existing phenotype.

        .. todo::
            Add checks for parent and code by making sure the relationships
            hold in the database.

            Also deleting the variable_type should not be allowed.

        """
        # Get the phenotype to make sure it exists.
        self.get_phenotype(name)

        updates = []
        for key, value in kwargs.items():
            if key not in PHENOTYPE_COLUMNS:
                raise TypeError("Unexpected column '{}' for phenotype."
                                "".format(key))
            elif value is None:
                # NULL values.
                updates.append("{}=NULL".format(key))
            elif key == "crf_page":
                # Integer.
                updates.append("{}={}".format(key, int(value)))
            else:
                # Strings.
                updates.append("{}='{}'".format(key, value))
        updates = ",".join(updates)

        self.cur.execute(
            "UPDATE phenotypes SET {} WHERE name=?".format(updates),
            (name, )
        )
        self.commit()

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
        """Set the sample IDs and order.

        This can take either an ndarray of dtype np.string_ or a regular Python
        array of str.

        """
        if type(samples) is np.ndarray:
            assert np.issubdtype(samples.dtype, np.string_)
        else:
            assert isinstance(samples[0], six.string_types), (
                "Sample IDs need to be strings."
            )
            samples = np.array(samples, dtype=np.string_)

        if self["frozen"] == "yes":
            raise FrozenCohortError()

        self.data.create_dataset("samples", data=samples)
        self._cache["samples"] = samples.astype(str)
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
        values = np.array(values)
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
            extra = ", ".join([str(i) for i in list(extra)[:5]])
            if len(extra) > 5:
                extra += ", ..."
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

    def _check_data_continuous(self, values, phenotype, _raise=False):
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
        n_values = inference.estimate_num_distinct(values)

        if n_values < 5:
            message = ("The phenotype '{}' is marked as continuous, but "
                       "it has a lot of redundancy. Perhaps it should be "
                       "modeled as a factor or another variable type."
                       "".format(phenotype))
            if _raise:
                raise ValueError(message)
            logger.warning(message)  # pragma: no cover

        # Outlier check.
        common_outlier = inference.find_overrepresented_outlier(values)
        if common_outlier is not None:
            message = ("The value '{}' is commonly found in the tails of the "
                       "distribution for '{}'. This could be because of bad "
                       "coding of missing values."
                       "".format(common_outlier, phenotype))
            if _raise:
                raise ValueError(message)
            logger.warning(message)  # pragma: no cover

    # Get information.
    def get_samples(self):
        """Get the ordered samples.

        :returns: An ordered list of samples in the manager.
        :rtype: np.ndarray

        """
        cached_samples = self._cache.get("samples")
        if cached_samples is not None:
            return cached_samples

        try:
            self._cache["samples"] = np.array(
                [i.decode("utf-8") for i in self.data["samples"]]
            )
            return self._cache["samples"]
        except KeyError:
            return None

    def get_phenotype(self, phenotype):
        """Get information on the phenotype."""
        self.cur.execute(
            "SELECT * FROM phenotypes WHERE name=?;",
            (phenotype, )
        )
        out = self.cur.fetchone()
        if out is None:
            raise KeyError("Could not find phenotype '{}'.".format(phenotype))
        out = dict(zip(PHENOTYPE_COLUMNS, out))

        return out

    def get_drug_users(self, drug_id, as_bool=False):
        """Return a boolean vector similar to a phenotype vector where 1
        represents drug users.

        If the sample is a user for a child or parent variable, it will be
        marked as a user for the provided drug.

        """
        samples = self.get_samples()
        v = np.zeros(len(samples), dtype=bool)

        # Get related drug IDs (parent or child).
        with ChEMBL() as chembl:
            related = chembl.get_related_drugs(drug_id)

        # Get all the samples that are users of the provided drug or related
        # drugs (e.g. parent or child).
        self.cur.execute(
            "SELECT sample_id FROM drug_users WHERE drug_id IN ({})".format(
                ",".join([str(i) for i in related])
            )
        )

        for sample in self.cur:
            # Get the index.
            v[samples == sample] = True

        if not as_bool:
            v = v.astype(float)

        return v

    def get_drug_users_atc(self, atc_code):
        """Returns a vector of drug users for drugs corresponding to an ATC
        code.

        """
        with ChEMBL() as db:
            drug_ids = db.get_drugs_with_atc(atc_code)
        return self._build_drug_user_vector(drug_ids)

    def get_drug_users_protein(self, uniprot_id, action=None):
        """Returns a vector of drug users for drugs modulating the protein
        represented by the provided ID.

        :param uniprot_id: The Uniprot ID (e.g. P31639)
        :type uniprot_id: str

        :param action: The action type. If None, any action_type is authorized.
                       Example action_types are: "INHIBITOR", "AGONIST" or
                       "ANTAGONIST".
        :type action: str

        :returns: A vector of drug users.
        :rtype: numpy.ndarray

        See: http://www.uniprot.org/

        """
        with ChEMBL() as db:
            drug_ids = db.get_drugs_modulating_protein(uniprot_id, action)
        return self._build_drug_user_vector(drug_ids)

    def _build_drug_user_vector(self, drug_ids):
        """Builds a vector of drug users for a list of drugs.

        This calls get_drug_users internally and ORs everything as this
        function is able to identify related (parent/child) drugs and mark
        users appropriately.

        """
        v = np.zeros(self.n, dtype=bool)
        for drug_id in drug_ids:
            v |= self.get_drug_users(drug_id, as_bool=True)
        return v.astype(float)

    def filter_phenotypes(self, missing_greater=None, missing_lower=None,
                          variable_type=None):
        """Get a list of phenotypes corresponding to the selected criterion.

        This method allows filtering of phenotypes with respect to:

        - The rate of missing values
        - The variable type

        """
        raise NotImplementedError()

    def contingency(self, phenotype1, phenotype2):
        """Build a contingency table for two discrete or factor phenotypes."""
        # Build an (m+1) x (n+1) matrix of counts and marginals.
        # m = n = 3 (missing, affected, unaffected) for discrete variables.
        meta1 = self.get_phenotype(phenotype1)
        meta2 = self.get_phenotype(phenotype2)
        dims = [0, 0]
        states = []
        labels = []
        for i, meta in enumerate((meta1, meta2)):
            if meta["variable_type"] == "discrete":
                dims[i] = 3  # NaN, 0, 1
                states.append([0, 1])
                labels.append(["missing", "control", "case"])
            elif meta["variable_type"] == "factor":
                code = self.get_code(meta["code_name"])

                dims[i] = len(code) + 1
                states.append([j[0] for j in code])
                labels.append(["missing"] + [j[1] for j in code])
            else:
                raise ValueError(
                    "Phenotype '{}' is not discrete or factor.".format(
                        meta["name"]
                    )
                )
        m, n = dims

        for i, phenotype in enumerate((phenotype1, phenotype2)):
            labels[i] = list(["".join(j) for j in zip(
                itertools.repeat("{} - ".format(phenotype)),
                labels[i]
            )])

        v1 = np.array(self.get_data(phenotype1, numpy=False))
        v2 = np.array(self.get_data(phenotype2, numpy=False))

        counts = np.zeros((m, n)).astype(int) - 1
        counts[0, 0] = np.sum(np.isnan(v1) & np.isnan(v2))
        for i in range(m - 1):
            counts[i + 1, 0] = np.sum(np.isnan(v2) & (v1 == states[0][i]))

            for j in range(n - 1):
                if i == 0:
                    counts[0, j + 1] = np.sum(
                        np.isnan(v1) & (v2 == states[1][j])
                    )
                counts[i + 1, j + 1] = np.sum(
                    (v1 == states[0][i]) &
                    (v2 == states[1][j])
                )

        return pd.DataFrame(counts, index=labels[0], columns=labels[1])

    def get_number_phenotypes(self):
        """Returns the number of phenotypes."""
        self.cur.execute("SELECT count(*) FROM phenotypes")
        return self.cur.fetchone()[0]

    def get_phenotypes_list(self, dummy=False):
        """Get a list of available phenotypes from the db."""
        # Creating the required SQL command
        sql = ("SELECT a.name FROM phenotypes a"
               " LEFT OUTER JOIN dummy_phenotypes b ON a.name=b.name"
               " WHERE b.name IS null;")
        if dummy:
            sql = "SELECT name FROM phenotypes;"

        self.cur.execute(sql)
        li = self.cur.fetchall()
        if li:
            li = [tu[0] for tu in li]
        return li

    def get_code(self, name):
        """Get the integer mappings for a given code.

        :param name: The code name to get from the database.
        :type name: str

        :returns: A list of tuples of key, value.
        :rtype: list:

        """
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
            return pd.Series(pd.Categorical.from_codes(
                data,
                [i[1] for i in sorted(code, key=lambda x: x[0])]
            ))

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

    def delete(self, phenotype):
        """Remove a phenotype from the manager."""
        self.cur.execute(
            "DELETE FROM phenotypes WHERE name=?", (phenotype, )
        )
        del self.data["data/{}".format(phenotype)]
        self.commit()

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
        for name in self.get_phenotypes_list():
            print(name)
            if name not in available:
                missing.add(name)

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

    def get_number_unaffected(self, phenotype):
        """Robust function to get the number of unaffected samples for a given
        phenotype.

        If the parent of a variable is unaffected (only valid for discrete
        variables), then it is unaffected, even though it will be marked as
        NA in the HDF5 container.

        .. warning::
            This does not take into account arbitrary factor values to identify
            unaffected individuals.

        """
        # Find the node from it's root.
        found = False
        for node in self.tree.depth_first_traversal():
            if node.data.name == phenotype:
                found = True
                break

        if not found:
            raise ValueError(
                "Phenotype '{}' is not in database.".format(phenotype)
            )

        # Walk back from the node to the root.
        cur = node
        path = collections.deque([cur])
        while cur.parent:
            path.appendleft(cur.parent)
            cur = cur.parent

        # Walk the path to find all the unaffected individuals as marked in the
        # hierarchy.
        unaffected = None
        for node in path:
            if node.data.variable_type == "discrete":
                # Get the data vector.
                data = self.get_data(node.data.name, numpy=True)
                if unaffected is None:
                    unaffected = (data == 0)
                else:
                    unaffected |= (data == 0)

        return np.sum(unaffected) if unaffected is not None else 0

    def get_number_missing(self, phenotype):
        """Get the true number of missing data points."""
        n_unaffected = self.get_number_unaffected(phenotype)
        meta = self.get_phenotype(phenotype)
        data = self.get_data(phenotype, numpy=True)

        if meta["variable_type"] == "discrete":
            return np.sum(np.isnan(data))
        elif meta["variable_type"] == "factor":
            nans = np.sum(data.isnull())
            return nans - n_unaffected
        elif meta["variable_type"] == "continuous":
            nans = np.sum(np.isnan(data))
            return nans - n_unaffected

    def variable(self, name):
        """Returns a variable object usable to create virtual phenotypes.

        Here are illustrative examples of how to use this given an instance
        of CohortManager (named `manager`)

        >>> v = manager.variable
        >>> age_at_event = v("yearOfEvent") - v("birthYear")
        array([25., 42., 70., 31.])
        >>> syndrome = (v("arrhythmia") | v("pacemaker")) & (v("age") < 50)
        >>> obese = (v("mass") / (v("height") ** 2)) > 30

        """
        data = self.get_data(name, numpy=True)
        meta = self.get_phenotype(name)
        if meta["variable_type"] not in ("continuous", "discrete"):
            raise TypeError("The virtual variable system can only be used "
                            "with continuous and discrete variables.")
        return _Variable(data)


class _Variable(object):
    """Building block to construct virtual phenotypes."""
    def __init__(self, data):
        self.data = data
        self.nans = np.isnan(self.data)

    def _discrete_comparison(self, a, b, function, nans=None):
        """Apply a comparison operator in discrete space."""
        # The cohort manager uses floats to represent discrete outcomes.
        values = function(a, b).astype(float)
        if nans is not None:
            values[nans] = np.nan
        return _Variable(values)

    def __gt__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         lambda a, b: a > b, self.nans)

    def __lt__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         lambda a, b: a < b, self.nans)

    def __ge__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         lambda a, b: a >= b, self.nans)

    def __le__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         lambda a, b: a <= b, self.nans)

    def __eq__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         lambda a, b: a == b, self.nans)

    def __ne__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         lambda a, b: a != b, self.nans)

    def _boolean_operation(self, a, b, function, nans="both"):
        """Apply a boolean operation in discrete space.

        For this function nans can be "both" or "any". If it is "any" NaNs will
        be propagated if they are observed in any of the two vectors.
        This is the desired bahavious for the AND operation.

        If nans is "both", then it is only propagated if both vectors have a
        NaN value at the given position. This is the desired behavious for
        OR.

        """
        # Check the types.
        vectors = [a, b]
        for i, vector in enumerate(vectors):
            if not isinstance(a, np.ndarray):
                raise TypeError("Boolean operations are only supported on "
                                "numpy arrays.")
            values = set(np.unique(vector[~np.isnan(vector)]))
            if values == {0, 1}:
                # This is the encoding we want.
                pass
            elif values == {True, False}:
                # Recode as floats.
                vectors[i] = vectors[i].astype(float)
            else:
                raise TypeError("Ambiguous encoding for boolean operations.")

        # Check size.
        if a.shape != b.shape:
            raise TypeError("Shape mismatch between a and b.")

        # Perform the comparison.
        if nans == "both":
            nans = np.isnan(a) & np.isnan(b)
        elif nans == "any":
            nans = np.isnan(a) | np.isnan(b)
        else:
            raise TypeError("Invalid value for the nan parameter. Use 'any' "
                            "or 'both'.")

        values = function(a, b).astype(float)
        values[nans] = np.nan
        return _Variable(values)

    def __and__(self, o):
        # Only works for two vectors that can be understood as discrete
        # variables.
        return self._boolean_operation(self.data, o.data,
                                       lambda a, b: (a == 1) & (b == 1),
                                       "any")

    def __rand__(self, o):
        return self.__and__(o)

    def __or__(self, o):
        return self._boolean_operation(self.data, o.data,
                                       lambda a, b: (a == 1) | (b == 1),
                                       "any")

    def __ror__(self, o):
        return self.__or__(o)

    def __sub__(self, o):
        o = o.data if isinstance(o, _Variable) else o
        return _Variable(self.data - o)

    def __add__(self, o):
        o = o.data if isinstance(o, _Variable) else o
        return _Variable(self.data + o)

    def __div__(self, o):
        o = o.data if isinstance(o, _Variable) else o
        return _Variable(self.data / o)

    __truediv__ = __div__

    def __mul__(self, o):
        o = o.data if isinstance(o, _Variable) else o
        return _Variable(self.data * o)

    def __pow__(self, a):
        if not (type(a) in (int, float)):
            raise TypeError("The power operator can only be used with "
                            "constant powers.")
        return _Variable(self.data ** a)

    def __invert__(self):
        if not self._is_discrete():
            raise TypeError("Can't invert non-discrete variable.")

        data = self.data.astype(float)
        data[data == 1] = 2
        data[data == 0] = 1
        data -= 1
        return _Variable(data)

    def __nonzero__(self):
        raise TypeError("The boolean 'not' operation is implemented using "
                        "the '~' operator.")

    def log(self):
        if self._is_discrete():
            raise TypeError(
                "'log' is only available for continuous variables."
            )
        return _Variable(np.log(self.data))

    def mean(self):
        if self._is_discrete():
            raise TypeError(
                "'mean' is only available for continuous variables."
            )
        return _Variable(np.nanmean(self.data))

    def std(self):
        if self._is_discrete():
            raise TypeError(
                "'std' is only available for continuous variables."
            )
        return _Variable(np.nanstd(self.data))

    def _is_discrete(self):
        observed = set(np.unique(self.data[~self.nans]))
        return observed == {0, 1}


def vector_map(data, _map):
    """Remaps numeric values in a data vector.

    :param data: A numpy array of integer or float dtype.
    :param _map: A list of tuple representing the mappings.
                 Alternatively, a dict can be provided directly.
                 ``[(2, 1), (1, 0), (3, -10)]`` would transform all the
                 ``2 -> 1``, ``1 -> 0`` and ``3 -> -10``.
    :returns: A remapped numpy array.

    The strategy used to avoid collisions when sequentially remapping is to
    use transitive mapping. This means that the mapping is done in two steps:
    A mapping to a unique (random) value and then a subsequent mapping to the
    target value.
    This strategy is only used if there are collisions.

    """

    if type(_map) is not dict:
        _map = dict(_map)

    # Check the dtype of the vector and the map.
    if np.issubdtype(data.dtype, int):
        source_dtype = int
    elif np.issubdtype(data.dtype, float):
        source_dtype = float
    else:
        raise TypeError("Invalid dtype: '{}'. This function only allows "
                        "int or float vectors as keys.".format(data.dtype))

    keys = set(_map.keys())
    targets = set(_map.values())

    # Infer the target dtype.
    def _safe_type(t):
        try:
            if np.isnan(t):
                return float
        except TypeError:
            pass
        return type(t)

    target_dtype = set([_safe_type(t) for t in targets])

    if len(target_dtype) != 1:
        raise TypeError("Ambiguous target dtype. Make sure that the provided "
                        "mapper uses consistent type for the second element "
                        "of the tuples.")
    target_dtype = target_dtype.pop()

    if target_dtype is int and source_dtype is float:
        raise TypeError("Remapping floats to integers is not possible "
                        "because of lost of data in type cast (this can "
                        "be fixed by using floats as the target values).")

    out = data.astype(target_dtype)

    for key, target in _map.items():
        if np.isnan(key):
            raise TypeError("Can't use NaNs as mapping keys.")

        if target in keys:
            # There will be a collision, so we need to use the transitive
            # mapping.
            transitive_key = hash(str(uuid.uuid4()))
            out[data == key] = transitive_key
            out[out == transitive_key] = target
        else:
            out[data == key] = target

    return out
