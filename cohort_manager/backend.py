"""
Backends for data storage.
"""


import os
import sqlite3
import logging

import numpy as np


logger = logging.getLogger(__name__)


# Numpy to SQLite lookups.
NP_INT_TYPES = (
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
    np.uint64
)
for _np_t in NP_INT_TYPES:
    sqlite3.register_adapter(_np_t, int)


class Backend(object):
    def connect(self, cohort_path):
        raise NotImplementedError()

    def log(self):
        logger.info("Using backend {}.".format(self.__class__.__name__))

    def add_data(self, variable, values):
        raise NotImplementedError()

    def delete_data(self, variable):
        raise NotImplementedError()

    def rename_variable(self, variable, new_name):
        raise NotImplementedError()

    def get_data(self, variable):
        raise NotImplementedError()

    def add_sample(self, sample_id):
        raise NotImplementedError()

    def delete_sample(self, sample_id):
        raise NotImplementedError()

    def set_samples(self, samples):
        raise NotImplementedError()

    def get_samples(self):
        raise NotImplementedError()

    def get_variables(self):
        raise NotImplementedError()

    @property
    def n():
        raise NotImplementedError()

    def close():
        raise NotImplementedError()


class FrozenCohortError(Exception):
    def __str__(self):
        return ("Once the sample order has been set, it is fixed and can't "
                "be changed.")


class UnknownSamplesError(Exception):
    def __init__(self, value=None):
        self.value = value

    def __str__(self):
        if self.value is None:
            # Default message.
            return ("Sample order was not set. It needs to be known before "
                    " data is added to enforce integrity checks.")
        return self.value


class SQLBasedBackend(Backend):
    def _create(self):
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS samples ("
            "  _cm_id INTEGER UNIQUE,"
            "  id TEXT UNIQUE,"
            "  PRIMARY KEY (_cm_id, id)"
            ")"
        )

        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS variables ("
            "  _cm_id INTEGER UNIQUE,"
            "  id TEXT UNIQUE,"
            "  PRIMARY KEY (_cm_id, id)"
            ")"
        )
        self.con.commit()

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    def set_samples(self, samples):
        # Make sure this has not been set before.
        self.cur.execute(
            "SELECT COUNT(id) FROM samples"
        )
        n = self.cur.fetchone()[0]
        if n != 0:
            raise FrozenCohortError()

        # We map samples to increasing integers to enforce ordering.
        self.cur.executemany(
            "INSERT INTO samples VALUES (?, ?)",
            zip(range(len(samples)), [str(s) for s in samples])
        )
        self.con.commit()

    @property
    def n(self):
        self.cur.execute("SELECT count(*) FROM samples")
        n = self.cur.fetchone()[0]
        return n if n != 0 else None

    def get_variables(self):
        self.cur.execute(
            "SELECT id FROM variables"
        )
        return {i[0] for i in self.cur}

    def get_samples(self):
        self.cur.execute(
            "SELECT _cm_id, id FROM samples ORDER BY _cm_id ASC"
        )
        out = [i[1] for i in self.cur]
        if not out:
            return None

        return np.array(out)

    def add_data(self, variable, values):
        # Check name collisions.
        if self._variable_exists(variable):
            raise ValueError(
                "Data for '{}' is already in the database.".format(variable)
            )

        # Check that the sample order has been set and is consistent with the
        # provided list.
        self.cur.execute("SELECT COUNT(id) FROM samples")
        n = self.cur.fetchone()[0]
        if n == 0:
            raise UnknownSamplesError()
        elif n != len(values):
            raise ValueError(
                "Expected {} values, got {} for variable '{}'."
                "".format(n, len(values), variable)
            )

        # Get the last variable id.
        self.cur.execute("SELECT MAX(_cm_id) FROM variables")
        i = self.cur.fetchone()[0]
        if i is None:
            i = 0  # First variable.
        else:
            i += 1

        # Register the variable (dynamic table names).
        if type(i) is not int:
            raise ValueError("Automatic keys should be integers.")

        table_name = "_variable_{}".format(i + 1)
        self.cur.execute(
            "INSERT INTO variables VALUES (?, ?)",
            (i, str(variable))
        )

        self.cur.execute(
            "CREATE TABLE {} ("
            "  _sample_id INTEGER REFERENCES samples(_cm_id),"
            "  value DOUBLE"
            ")".format(table_name)
        )
        self.con.commit()

        # Insert the data.
        self.cur.execute("SELECT _cm_id FROM samples ORDER BY _cm_id ASC")
        samples = [i[0] for i in self.cur]

        self.cur.executemany(
            "INSERT INTO {} VALUES (?, ?)".format(table_name),
            zip(samples, values)
        )
        self.con.commit()

    def _get_variable_table(self, variable):
        # Get the id.
        self.cur.execute(
            "SELECT _cm_id, id FROM variables WHERE id=?",
            (variable, )
        )
        tu = self.cur.fetchone()
        if tu is None:
            raise KeyError("No data for '{}'.".format(variable))

        return "_variable_{}".format(tu[0] + 1)

    def delete_data(self, variable):
        table_name = self._get_variable_table(variable)
        self.cur.execute("DROP TABLE {}".format(table_name))
        self.cur.execute(
            "DELETE FROM variables WHERE id=?", (variable, )
        )
        self.con.commit()

    def get_data(self, variable):
        table_name = self._get_variable_table(variable)
        self.cur.execute(
            "SELECT _sample_id, value "
            "FROM {} ORDER BY _sample_id ASC".format(table_name)
        )
        return np.array([i[1] for i in self.cur], dtype=float)

    def delete_sample(self, sample_id):
        # Find the sample _cm_id
        self.cur.execute(
            "SELECT _cm_id FROM samples WHERE id=?",
            (sample_id, )
        )
        id = self.cur.fetchone()

        if not id:
            raise ValueError("Unknown sample '{}'.".format(sample_id))

        id = id[0]

        # Delete from all the data tables.
        for table_name in self._data_tables():
            self.cur.execute(
                "DELETE FROM {} WHERE _sample_id=?".format(table_name),
                (id, )
            )

        # Delete from the samples table.
        self.cur.execute("DELETE FROM samples WHERE _cm_id=?", (id, ))
        self.con.commit()

    def _data_tables(self):
        self.cur.execute("SELECT _cm_id, id FROM variables")
        for table_id, variable in self.cur.fetchall():
            yield "_variable_{}".format(table_id + 1)

    def add_sample(self, sample_id):
        # Find the next sample _cm_id.
        self.cur.execute(
            "SELECT MAX(_cm_id) FROM samples"
        )
        id = self.cur.fetchone()

        if not id:
            raise ValueError(
                "Adding samples is meant to be used on cohorts whose sample "
                "order has already been set."
            )
        id = id[0] + 1  # Increment for the new sample.

        # Make sure there are no ID collisions.
        if self._sample_exists(sample_id):
            raise ValueError(
                "Sample named '{}' already exists.".format(sample_id)
            )

        # Add the sample.
        self.cur.execute("INSERT INTO samples VALUES (?, ?)", (id, sample_id))

        # Add to the data tables.
        for table_name in self._data_tables():
            self.cur.execute(
                "INSERT INTO {} VALUES (?, NULL)".format(table_name),
                (id, )
            )

        self.con.commit()

    def _sample_exists(self, id):
        self.cur.execute("SELECT * FROM samples WHERE id=?", (id, ))
        tu = self.cur.fetchone()
        return tu is not None

    def _variable_exists(self, name):
        self.cur.execute("SELECT * FROM variables WHERE id=?", (name, ))
        tu = self.cur.fetchone()
        return tu is not None

    def rename_variable(self, old, new):
        # Check that old name exists.
        if not self._variable_exists(old):
            raise KeyError("No data for '{}'.".format(old))

        if self._variable_exists(new):
            raise ValueError("New variable name already exists.")

        self.cur.execute(
            "UPDATE variables SET id=? WHERE id=?", (new, old)
        )
        self.con.commit()


class SQLiteBackend(SQLBasedBackend):
    def connect(self, path):
        self.filename = os.path.join(path, "data.db")
        self.con = sqlite3.connect(self.filename)
        self.cur = self.con.cursor()

        self._create()

    def log(self):
        logger.info("Using backend {} connected to '{}'.".format(
            self.__class__.__name__, self.filename
        ))


def backend_from_string(s):
    choices = {
        "SQLiteBackend": SQLiteBackend,
    }

    if s not in choices:
        raise ValueError(
            "Unknown backend '{}'. Choices are: {}."
            "".format(s, ", ".join(choices.keys()))
        )

    else:
        return choices[s]()


def sparse_to_dense(affected_samples, samples):
    """Convert a sparse representation to a dense vector.

    :param affected_samples: A list of samples that should be set to '1' in the
                             dense representation.
    :type affected_samples: list

    :param samples: The ordered samples list for the dense vector.
    :type samples: list

    For example, if s1 and s3 are affected and the sample order is [s1, s2, s3,
    s4, s5]:

        v = sparse_to_dense(["s1", "s3"], ["s1", "s2", "s3", "s4", "s5"])
        # v = np.array([1, 0, 1, 0, 0])

    """
    v = np.zeros(len(samples), dtype=float)

    # Build id to index dict.
    d = {sample: idx for idx, sample in enumerate(samples)}

    for sample in affected_samples:
        if sample in d:
            v[d[sample]] = 1
        else:
            logger.warning(
                "Could not find sample '{}' in the samples list corresponding "
                "to sparse vector.".format(sample)
            )
    return v
