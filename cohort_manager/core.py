import logging
import sqlite3
import os

import six
import h5py

"""
"""

logger = logging.getLogger(__name__)


PHENOTYPE_COLUMNS = ("name", "icd10", "parent", "variable_type", "crf_page",
                     "question", "code_name")


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
        return ("Once the sample order has been set, the database becomes "
                "immutable.")


class CohortManager(object):
    def __init__(self, name, path=None):
        self.name = name
        self.path = path if path else os.path.abspath(name)
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self._discover_install()

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
        self.con.commit()
        self.con.close()

    # Public API.
    @property
    def n(self):
        if self["frozen"] == "yes":
            return self.get_samples().shape[0]
        return None

    def commit(self):
        self.con.commit()

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

    def add_data(self, vector):
        """Insert a data vector."""
        raise NotImplementedError()

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
        # TODO Check if information on the variable is already in the database.
        if self["frozen"] == "yes":
            n = self.n
            if len(values) != n:
                raise ValueError(
                    "Expected {} values, got {}.".format(n, len(values))
                )

            if phenotype in self.data["data"].keys():
                raise ValueError("Data for '{}' is already "
                                 "in the database.".format(phenotype))

            self.data["data"].create_dataset(phenotype, data=values)
        else:
            raise UnknownSamplesError()

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

    def get_data(self, phenotype):
        """Get a phenotype vector."""
        try:
            return self.data["data/" + str(phenotype)]
        except KeyError:
            return None

    def get_code_names(self):
        """Get a set of code names."""
        self.cur.execute(
            "SELECT DISTINCT name FROM code;"        
        )
        return set([i[0] for i in self.cur.fetchall()])
