"""
"""

from __future__ import division

import operator
import itertools
import logging
import sqlite3
import atexit
import datetime
import uuid
import os

import six
import h5py
import numpy as np
import pandas as pd
from six.moves import range

from .phenotype_tree import PHENOTYPE_COLUMNS, tree_from_database
from . import types
from .drugs.chembl import ChEMBL
from .drugs.atc import get_atc_code_level


logger = logging.getLogger(__name__)
DRUG_EXTRA_FIELDS = (
    "start_date", "end_date", "indication", "dose", "dose_unit"
)


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
        data = self._manager.get_data(phenotype)
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
    """The Manager class which contains all the core API methods.

    This is the main way of interacting with the data from the CohortManager.

    """
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
            " start_date INTEGER,"
            " end_date INTEGER,"
            " indication TEXT,"
            " dose REAL,"
            " dose_unit TEXT"
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
        if not types.is_type_name(kwargs["variable_type"]):
            raise TypeError("Unknown variable type '{}'.".format(
                kwargs["variable_type"]
            ))

        # Factor variables need a code_name.
        if types.type_str(kwargs["variable_type"]).subtype_of(types.Factor):
            if kwargs.get("code_name") is None:
                raise TypeError("Factor variables need a 'code_name'.")

        self.cur.execute(
            "INSERT INTO phenotypes VALUES (?, ?, ?, ?, ?, ?, ?)",
            tuple(values)
        )
        self.commit()

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

    def register_drug_user(self, drug_id, sample, **kwargs):
        """Register that 'sample' is a user of drug 'drug_id'.

        :param drug_id: A valid ChEMBL molregno.
        :type drug_id: int

        :param sample: The sample ID.
        :type sample: str

        Other parameters are also allowed:
            - start_date: (datetime) Recorded date of therapy start.
            - end_date: (datetime) The end date.
            - indication: (str) The reason why the individual uses the drug.
                                This is unstructured for now.
            - dose: (float) The medication dose.
            - dose_unit: (str) The units for the dose (e.g. mg).

        """
        # Clean data from the kwargs.
        self._check_drug_fields(kwargs)

        # Format date if date object was passed.
        for field in ("start_date", "end_date"):
            if type(kwargs.get(field)) is datetime.date:
                kwargs[field] = kwargs[field].strftime("%Y-%m-%d")

        # Check that start < end.
        if kwargs.get("start_date") and kwargs.get("end_date"):
            strptime = datetime.datetime.strptime
            start = strptime(kwargs["start_date"], "%Y-%m-%d")
            end = strptime(kwargs["end_date"], "%Y-%m-%d")

            if end < start:
                raise ValueError("end_date is before than start date.")

        extra_params = map(kwargs.get, DRUG_EXTRA_FIELDS)

        # Check that the sample is valid.
        if self.get_samples() is not None and sample not in self.get_samples():
            raise ValueError("Sample '{}' not in the manager.".format(sample))

        self.cur.execute(
            "INSERT INTO drug_users VALUES (?, ?, ?, ?, ?, ?, ?)",
            tuple(itertools.chain((str(drug_id), sample), extra_params))
        )

    def _check_drug_fields(self, fields):
        bad_date_message = (
            "Invalid date representation for '{field}'. Use ISO 8601 (e.g. "
            "2010-02-26)."
        )

        bad_dose_message = "Invalid dose (dose should be a number)."

        for field in ("start_date", "end_date"):
            if field in fields:
                val = fields[field]
                if (type(val) is datetime.date or val is None):
                    continue

                if not types.Date.predicate(val):
                    raise ValueError(bad_date_message.format(field=field))

        try:
            if "dose" in fields:
                float(fields["dose"])
        except ValueError:
            raise ValueError(bad_dose_message)

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

        This can take either an ndarray of dtype `np.string_` or a regular
        Python array of str.

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
        try:
            meta = self.get_phenotype(phenotype)
        except KeyError:
            meta = None

        if meta is None:
            raise ValueError(
                "Could not find metadata for '{}'. Add the phenotype before "
                "binding data.".format(phenotype)
            )

        variable_type = types.type_str(meta["variable_type"])
        code_name = meta.get("code_name")

        if self["frozen"] != "yes":
            raise UnknownSamplesError()

        n = self.n
        if len(values) != n:
            raise ValueError(
                "Expected {} values, got {}.".format(n, len(values))
            )

        if phenotype in self.data["data"].keys():
            raise ValueError("Data for '{}' is already in the database."
                             "".format(phenotype))

        # Because checking for factor variables is a bit different, we do
        # additional checks.
        # Otherwise, we just pass the data to the relevant
        # check method as defined in the types module.
        if variable_type.subtype_of(types.Factor):
            # Get the factor mapping.
            self.cur.execute(
                "SELECT key, value FROM code WHERE name=?",
                (code_name, )
            )
            mapping = self.cur.fetchall()

            if not mapping:
                raise CohortDataError("Could not find the code for factor "
                                      "variable '{}'. The code name is "
                                      "'{}'.".format(phenotype, code_name))

            _type_check(phenotype, values, variable_type, mapping)

        else:
            _type_check(phenotype, values, variable_type)

        # If no exception was raised, store the data.
        self.data["data"].create_dataset(
            phenotype,
            data=variable_type.encode(values)
        )

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

    def is_valid_phenotype(self, phenotype):
        """Checks if a phenotype exists."""
        self.cur.execute(
            "SELECT name FROM phenotypes WHERE name=?;", (phenotype, )
        )
        return self.cur.fetchone() is not None

    def is_dummy_phenotype(self, phenotype):
        """Checks if a phenotype is actually a dummy phenotype."""
        self.cur.execute(
            "SELECT name FROM dummy_phenotypes WHERE name=?;", (phenotype, )
        )
        return self.cur.fetchone() is not None

    def _build_drug_filtering_sql(self, filters):
        """Builds SQL statements to filter drug users as described in the
        `get_drug_users` method.

        """
        known_filters = ("between_dates", "indication", "dose")

        extra = set(filters.keys()) - set(known_filters)
        if extra:
            raise TypeError("Unknown filter(s) '{}'.".format(extra))

        sql = []
        args = []

        # Apply the date filter.
        if "between_dates" in filters.keys():
            dates = filters["between_dates"]
            if type(dates) is not tuple or len(dates) != 2:
                raise ValueError("Expected the 'between_dates' filter to be a "
                                 "tuple of date str.")

            start, end = dates
            dates = [start, end]
            for i in (0, 1):
                if isinstance(dates[i], datetime.date):
                    dates[i] = dates[i].strftime("%Y-%m-%d")
                elif not (dates[i] is None or types.Date.predicate(dates[i])):
                    raise ValueError("Invalid date '{}' (not ISO 8601)."
                                     "".format(dates[i]))

            if start is None and end is None:
                raise ValueError("Provide at least one boundary for "
                                 "between_dates.")

            if start is not None:
                sql.append("end_date>?")
                args.append(start)

            if end is not None:
                sql.append("start_date<?")
                args.append(end)

        # Apply the indication filter.
        if filters.get("indication"):
            sql.append("LOWER(indication) LIKE ?")
            args.append(filters["indication"].lower())

        # Apply the dose filter.
        if "dose" in filters.keys():
            dose = filters["dose"]
            if (type(dose) is not tuple) or len(dose) > 2:
                raise ValueError("Expected the 'dose' filter to be a tuple of "
                                 "dose (float) and unit (str, optional).")

            if len(dose) == 2:
                dose, unit = dose
            else:
                dose = dose[0]
                unit = None

            dose = float(dose)

            sql.append("dose=?")
            args.append(dose)

            if unit:
                sql.append("dose_unit=?")
                args.append(unit)

        sql = " AND ".join(sql)
        return sql, tuple(args)

    def get_drug_users(self, drug_id, as_bool=False, **filters):
        """Return a boolean vector similar to a phenotype vector where 1
        represents drug users.

        If the sample is a user for a child or parent variable, it will be
        marked as a user for the provided drug.

        It is also possible to use filters:
            - between_dates (tuple): Only returns users between the provided
                                     dates. It is possible to use None to set
                                     only a lower or upper bound.
            - indication (str): Filter for a specific indication.
            - dose (tuple): A tuple of dose (float) and unit (str). The unit
                            can be set to None.


        The indication match will always be case insensitive. SQL wildcards
        are authorized ('%').

        TODO: It could be practical to have dose_greater and dose_lower
        filters.

        """
        where_clause, args = self._build_drug_filtering_sql(filters)

        samples = self.get_samples()
        v = np.zeros(len(samples), dtype=bool)

        # Get related drug IDs (parent or child).
        with ChEMBL() as chembl:
            related = chembl.get_related_drugs(drug_id)

        # Get all the samples that are users of the provided drug or related
        # drugs (e.g. parent or child).
        sql = "SELECT sample_id FROM drug_users WHERE drug_id IN ({})".format(
            ",".join([str(i) for i in related])
        )

        if where_clause:
            sql += " AND {}".format(where_clause)
            self.cur.execute(sql, args)
        else:
            self.cur.execute(sql)

        for sample in self.cur:
            # Get the index.
            v[samples == sample] = True

        if not as_bool:
            v = v.astype(float)

        return v

    def get_drug_users_atc(self, atc_code, **filters):
        """Returns a vector of drug users for drugs corresponding to an ATC
        code.

        """
        with ChEMBL() as db:
            drug_ids = db.get_drugs_with_atc(atc_code)
        return self._build_drug_user_vector(drug_ids, **filters)

    def get_drug_users_protein(self, uniprot_id, action=None, **filters):
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
        return self._build_drug_user_vector(drug_ids, **filters)

    def _build_drug_user_vector(self, drug_ids, **filters):
        """Builds a vector of drug users for a list of drugs.

        This calls get_drug_users internally and ORs everything as this
        function is able to identify related (parent/child) drugs and mark
        users appropriately.

        """
        v = np.zeros(self.n, dtype=bool)
        for drug_id in drug_ids:
            v |= self.get_drug_users(drug_id, as_bool=True, **filters)
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
            t = meta["variable_type"]
            if types.type_str(t).subtype_of(types.Discrete):
                dims[i] = 3  # NaN, 0, 1
                states.append([0, 1])
                labels.append(["missing", "control", "case"])
            elif types.type_str(t).subtype_of(types.Factor):
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

        v1 = np.array(self._get_raw_data(
            phenotype1,
            cm_type=types.type_str(meta1["variable_type"])
        ))
        v2 = np.array(self._get_raw_data(
            phenotype2,
            cm_type=types.type_str(meta2["variable_type"])
        ))

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
        """Get a list of available phenotypes from the db.

        :param dummy: A flag to return the dummy phenotypes or not.
        :type dummy: bool

        """
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

    def _get_raw_data(self, phenotype, cm_type):
        """Get the non-processed data.

        The only data manipulation is the type-level decode (which, by
        default, converts the dataset to a numpy array).

        """
        try:
            data = self.data["data/{}".format(phenotype)]
            return cm_type.decode(data)
        except KeyError:
            raise KeyError("No data for '{}'.".format(phenotype))

    def get_data(self, phenotype):
        """Get a phenotype vector as a numpy array."""
        # Get metadata.
        meta = self.get_phenotype(phenotype)
        t = meta["variable_type"]

        # Make sure the type is valid.
        if not types.is_type_name(t):
            raise CohortDataError(
                "Data for variable '{}' was stored, but it is of an unknown "
                "type: {}.".format(phenotype, t)
            )

        # Discrete phenotypes are explicitly recoded to account for unaffected
        # parents.
        if types.type_str(t).subtype_of(types.Discrete):
            return self._get_discrete_data(phenotype)

        data = self._get_raw_data(phenotype, cm_type=types.type_str(t))

        # Factor variables are returned as pandas series.
        if types.type_str(t).subtype_of(types.Factor):
            return self._represent_factor_data(data, meta)

        # Any other variable types are returned as is.
        return data

    def _check_unaffected_parent_variables(self, phenotype):
        unaffected = np.full(self.n, False, dtype=bool)
        cur = self.get_phenotype(phenotype)["parent"]

        while cur is not None:
            meta = self.get_phenotype(cur)
            _type = types.type_str(meta["variable_type"])
            if _type.subtype_of(types.Discrete):
                data = self._get_raw_data(cur, cm_type=types.Discrete)
                unaffected[data == 0] = True

            cur = meta["parent"]

        return unaffected

    def _get_discrete_data(self, phenotype):
        unaffected = self._check_unaffected_parent_variables(phenotype)
        data = self._get_raw_data(phenotype, cm_type=types.Discrete)
        data[unaffected] = 0
        return data

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

    def delete(self, phenotype, _db_only=False):
        """Remove a phenotype from the manager."""
        # Checking the phenotype is valid
        if not self.is_valid_phenotype(phenotype):
            raise ValueError("Invalid phenotype '{}'. It doesn't exist in the "
                             "database.".format(phenotype))

        # Checking if the phenotype is a dummy phenotype
        is_dummy = self.is_dummy_phenotype(phenotype)

        # Deleting the entry
        self.cur.execute(
            "DELETE FROM phenotypes WHERE name=?", (phenotype, )
        )

        # Deleting the entry in the dummy phenotypes table
        if is_dummy:
            self.cur.execute(
                "DELETE FROM dummy_phenotypes WHERE name=?", (phenotype, )
            )

        # Deleting the data if the phenotype is not a dummy one
        if not (is_dummy or _db_only):
            del self.data["data/{}".format(phenotype)]

        # Committing
        self.commit()

    def rename(self, old_name, new_name):
        self.get_phenotype(old_name)  # Will raise KeyError if need be.
        new_name_is_free = False
        try:
            self.get_phenotype(new_name)
        except KeyError:
            new_name_is_free = True

        if not new_name_is_free:
            raise ValueError("New phenotype name already exists.")

        self.cur.execute(
            "UPDATE phenotypes SET name=? WHERE name=?", (new_name, old_name)
        )

        # Update the name in the data.
        new_name = "data/{}".format(new_name)
        old_name = "data/{}".format(old_name)
        self.data[new_name] = self.data[old_name]
        del self.data[old_name]

        self.commit()

    def merge_as_factor(self, new_name, phenotypes, delete=False):
        """Merge discrete phenotypes into a single factor variable.

        # Create the 'my_var' variable by combining the discrete phenotypes
        # A, B and C.
        >>> manager.merge_as_factor("my_var", ("A", "B", "C"), delete=True)

        """
        # Prepare the data and check type.
        data = np.full(self.n, np.nan, dtype=np.float)
        filled = np.zeros(self.n, dtype=bool)
        for i, phenotype in enumerate(phenotypes):
            # Get phenotype information.
            info = self.get_phenotype(phenotype)
            t = info["variable_type"]
            if not types.type_str(t).subtype_of(types.Discrete):
                raise ValueError("Can't merge non-discrete variable ({}) into "
                                 "factor.".format(phenotype))

            # Add the data.
            mask = self.get_data(phenotype) == 1.0
            if np.any(filled & mask):
                raise ValueError(
                    "Some discrete variables in the provided set are not "
                    "mutually exclusive. They can't be combined as a single "
                    "factor variable."
                )
            filled[mask] = True
            data[mask] = i

        # Create the code.
        code_name = str(uuid.uuid4()).split("-")[0]
        for i, phenotype in enumerate(phenotypes):
            self.add_code(code_name, i, phenotype)

        # Create the variable.
        self.add_phenotype(name=new_name, variable_type="factor",
                           code_name=code_name)

        # Fill the data.
        self.add_data(new_name, data)

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

    def check_phenotypes_have_data(self, printer):
        """Check that all the phenotypes in the database have data entries."""
        logger.info("Making sure that data is available for all phenotypes "
                    "in the database.")
        missing = set()
        available = set(self.data["data"].keys())
        for name in self.get_phenotypes_list():
            if name not in available:
                missing.add(name)

        if missing:
            printer("Missing data for phenotypes '{}'.".format(missing))
            return False
        logger.debug("Phenotype in the database are consistent with the "
                     "binary data store.")
        return True

    def get_number_missing(self, phenotype):
        """Get the true number of missing data points."""
        meta = self.get_phenotype(phenotype)
        data = self.get_data(phenotype)

        t = meta["variable_type"]

        # For discrete data, unaffected individuals are automatically
        # reclassified.
        if types.type_str(t).subtype_of(types.Discrete):
            return np.sum(np.isnan(data))

        n_unaffected = np.sum(
            self._check_unaffected_parent_variables(phenotype)
        )

        # Dates and Factors are returned as pandas series.
        encoded_as_series = (
            types.type_str(t).subtype_of(types.Factor) or
            types.type_str(t).subtype_of(types.Date)
        )
        if encoded_as_series:
            nans = np.sum(data.isnull())
            return nans - n_unaffected

        # Generic case.
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
        data = self.get_data(name)
        meta = self.get_phenotype(name)
        t = types.type_str(meta["variable_type"])
        if not (t.subtype_of(types.Continuous) or
                t.subtype_of(types.Discrete)):
            raise TypeError("The virtual variable system can only be used "
                            "with continuous and discrete variables.")
        return _Variable(data)

    def drug(self, drug_code, **filters):
        """Returns a variable object representing drug user status.

        >>> v = manager.variable
        >>> drug = manager.drug
        >>> # This will take beta-blocker users that had a previous MI.
        >>> outcome = drug("C07") & v("MI")

        This is compatible with the virtual variable interface.

        """
        # The drug code should be an ATC code or an integer.
        is_atc = False

        if not isinstance(drug_code, int):
            try:
                get_atc_code_level(drug_code)
                is_atc = True
            except ValueError:
                is_atc = False

        if is_atc:
            drug_users = self.get_drug_users_atc(drug_code, **filters)

        else:
            try:
                drug_code = int(drug_code)
            except ValueError:
                raise ValueError("Invalid drug code '{}'. Drug codes should "
                                 "be either an ATC code or a ChEMBL molecule "
                                 "id (molregno).".format(drug_code))
            drug_users = self.get_drug_users(drug_code, **filters)

        return _Variable(drug_users)


class _Variable(object):
    """Building block to construct virtual phenotypes."""
    def __init__(self, data):
        self.data = data

    @staticmethod
    def _discrete_comparison(a, b, function):
        """Apply a comparison operator in discrete space."""
        nans = np.isnan(a)
        if type(b) is np.ndarray:
            assert a.shape == b.shape
            assert len(a.shape) == 1
            nans |= np.isnan(b)
            vals = np.where(~nans)[0]
            b = b[vals]
        else:
            vals = np.where(~nans)[0]

        out = np.full(a.shape[0], np.nan)
        out[vals] = function(a[vals], b)

        return _Variable(out)

    def __gt__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         operator.gt)

    def __lt__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         operator.lt)

    def __ge__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         operator.ge)

    def __le__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         operator.le)

    def __eq__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         operator.eq)

    def __ne__(self, o):
        return self._discrete_comparison(self.data, getattr(o, "data", o),
                                         operator.ne)

    def __and__(self, o):
        """And (&) operator.

        +---+---+---+---+
        |a/b| 0 | 1 | ø |
        +---+---+---+---+
        | 0 | 0 | 0 | 0 |
        +---+---+---+---+
        | 1 | 0 | 1 | ø |
        +---+---+---+---+
        | ø | 0 | ø | ø |
        +---+---+---+---+

        """
        v = np.zeros(self.data.shape[0], dtype=float)
        v[(self.data == 1) & (o.data == 1)] = 1
        v[np.isnan(self.data) & (o.data != 0)] = np.nan
        v[(self.data != 0) & np.isnan(o.data)] = np.nan
        return _Variable(v)

    def __or__(self, o):
        """OR (|) operator.

        Truth table:

        +---+---+---+---+
        |a/b| 0 | 1 | ø |
        +---+---+---+---+
        | 0 | 0 | 1 | ø |
        +---+---+---+---+
        | 1 | 1 | 1 | 1 |
        +---+---+---+---+
        | ø | ø | 1 | ø |
        +---+---+---+---+

        Unit tests are based on this table.

        """
        v = np.full(self.data.shape[0], np.nan)
        v[(self.data == 1) | (o.data == 1)] = 1
        v[(self.data == 0) & (o.data == 0)] = 0
        return _Variable(v)

    def __xor__(self, o):
        """The XOR (^) operator.

        Truth table:

        +---+---+---+---+
        |a/b| 0 | 1 | ø |
        +---+---+---+---+
        | 0 | 0 | 1 | ø |
        +---+---+---+---+
        | 1 | 1 | 0 | ø |
        +---+---+---+---+
        | ø | ø | ø | ø |
        +---+---+---+---+

        """
        v = np.full(self.data.shape[0], np.nan)
        v[(self.data == 0) & (o.data == 0)] = 0
        v[(self.data == 1) & (o.data == 1)] = 0
        v[(self.data == 1) & (o.data == 0)] = 1
        v[(self.data == 0) & (o.data == 1)] = 1
        return _Variable(v)

    def __sub__(self, o):
        o = o.data if isinstance(o, _Variable) else o
        return _Variable(self.data - o)

    def __rsub__(self, o):
        return self.__sub__(o)

    def __add__(self, o):
        o = o.data if isinstance(o, _Variable) else o
        return _Variable(self.data + o)

    def __radd__(self, o):
        return self.__add__(o)

    def __div__(self, o):
        o = o.data if isinstance(o, _Variable) else o
        return _Variable(self.data / o)

    def __rdiv__(self, o):
        return self.__div__(o)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __mul__(self, o):
        o = o.data if isinstance(o, _Variable) else o
        return _Variable(self.data * o)

    def __rmul__(self, o):
        return self.__mul__(o)

    def __pow__(self, a):
        if not (type(a) in (int, float)):
            raise TypeError("The power operator can only be used with "
                            "constant powers.")
        return _Variable(self.data ** a)

    def __invert__(self):
        if not self._is_discrete():
            raise TypeError("Can't invert non-discrete variable (or "
                            "case-only).")

        data = self.data.astype(float)
        data[data == 0] = 2
        data -= 1
        return _Variable(data)

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

    def _pass_filter(self, threshold, mode):
        if self._is_discrete():
            raise TypeError(
                "'low_pass' is only available for continuous variables."
            )
        data = self.data.copy()
        op = operator.le if mode == "low" else operator.ge
        for i in range(data.shape[0]):
            if not np.isnan(data[i]):
                data[i] = data[i] if op(data[i], threshold) else np.nan

        return _Variable(data)

    def low_pass(self, threshold):
        return self._pass_filter(threshold, "low")

    def high_pass(self, threshold):
        return self._pass_filter(threshold, "high")

    def std(self):
        if self._is_discrete():
            raise TypeError(
                "'std' is only available for continuous variables."
            )
        return _Variable(np.nanstd(self.data))

    def _is_discrete(self):
        observed = set(np.unique(self.data[~np.isnan(self.data)]))
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


def _type_check(name, values, _type, *check_args):
    """Wraps type checks to provide better logging."""
    exception = None
    try:
        _type.check(values, *check_args)
    except types.InvalidValues as e:
        exception = e
        logger.warning("Variable '{}' failed data type validation checks and "
                       "cannot be inserted.".format(name))
    if exception is not None:
        raise exception
