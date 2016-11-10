"""
Parse the Excel file used to build the cohort.
"""

import logging
import json

import pandas as pd
import numpy as np

from .core import CohortManager
from . import types


logger = logging.getLogger(__name__)


class InvalidConfig(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Invalid configuration file.\n{}".format(self.value)


def import_file(filename, cohort_name):
    """Fill the database using the provided Excel configuration file."""

    # Create a CohortManager.
    manager = CohortManager(cohort_name)

    # Read the file.
    data = pd.read_excel(filename, sheetname=None)  # Dict of DataFrames
    metadata = data["Metadata"]
    metadata = {i: j for i, j in metadata.values}
    # Known missings is a list.
    metadata["known_missings"] = json.loads(metadata["known_missings"])

    variables = data["Variables"]
    variables.index = range(2, variables.shape[0] + 2)

    # Check variables data for name conflicts.
    variables = _clean_global_file(variables)

    # Start parsing individual files.
    for filename in variables["path"].unique():
        file_variables = variables.loc[
            variables["path"] == filename, :
        ]
        data = _read_and_clean(filename, file_variables, metadata)
        if data is None:
            logger.debug("SKIPPING '{}'".format(filename))
            continue

        _do_import(manager, file_variables, data)


def _clean_global_file(variables):
    # Remove variables if "import" is False.
    variables = variables.loc[variables["import"], :]

    # Check for name conflicts.
    dups = variables.duplicated(subset="database_name", keep=False)

    # Ignore the name conflicts in the sample IDs.
    dups = dups & (~ (variables["variable_type"] == "unique_key"))

    if dups.any():
        dups_list = list(set(variables["database_name"][dups]))

        logger.warning(
            "Some variables will NOT be imported because they have duplicated "
            "names: {}.".format(", ".join(_quote_li(dups_list)))
        )

        # Remove the duplicates.
        variables = variables.loc[~dups, :]

    # Check for missing required information.
    required_cols = ["column_name", "database_name", "path"]
    nulls = variables[required_cols].isnull()
    if np.any(nulls.values):
        null_rows = np.any(nulls.values, axis=1)
        # Convert to excel index.
        excel_null_rows = [str(i) for i in nulls.iloc[null_rows, :].index]

        logger.warning(
            "Some variables will NOT be imported because they are missing "
            "data in at least one of the mandatory columns ({}) in the "
            "following rows: {}.".format(
                ", ".join(_quote_li(required_cols)),
                ", ".join(excel_null_rows)
            )
        )

        # Remove rows with missing data.
        variables = variables.dropna(axis=0, subset=required_cols)

    return variables


def _read_and_clean(filename, variables, metadata):
    # Check that there is a sample_id column.
    key = np.where(variables["variable_type"] == "unique_key")[0]

    if len(key) < 1:
        logger.critical(
            "The file '{}' WILL NOT BE IMPORTED because there was no variable "
            "with the 'unique_key' type which is used to represent the column "
            "containing sample IDs.".format(filename)
        )
        return

    elif len(key) > 1:
        logger.critical(
            "The file '{}' WILL NOT BE IMPORTED because there are multiple "
            "variables with the 'unique_key' type which is used to represent "
            "the (single) column containing sample IDs.".format(filename)
        )
        return

    key = key[0]

    # Read the file and set the index.
    data = pd.read_csv(
        filename,
        delimiter=metadata["delimiter"],
        usecols=list(variables["column_name"]),
        na_values=metadata["known_missings"]
    )

    key_name = variables.iloc[key, :]["column_name"]

    data[key_name] = data[key_name].astype(str)
    try:
        data.set_index(key_name, verify_integrity=True, inplace=True)
    except ValueError:
        logger.critical(
            "The file '{}' WILL NOT BE IMPORTED because the index column "
            "('{}') contains duplicates.".format(filename, key_name)
        )
        return

    return data


def _do_import(manager, variables, data):
    # Check if the manager has a sample list.
    samples = manager.get_samples()
    if samples is None:
        # No sample order has been set.
        manager.set_samples(data.index.values)

    else:
        diff = set(data.index) - set(manager.get_samples())

        # Check if there are new samples.
        if len(diff) > 0:
            # Add the new samples.
            for sample in diff:
                manager.add_sample(sample)

        # Build an empty DataFrame indexed by the current order.
        df = pd.DataFrame({"sample_id": manager.get_samples()})
        df.set_index("sample_id", inplace=True)

        # Represent the data with respect to the manager's order.
        data = df.join(data)

    # Import individual variables.
    code_name = None  # This will get set for factors.
    for tu in variables.itertuples():
        if tu.variable_type == "unique_key":
            continue

        mask = ~ data[tu.column_name].isnull().values
        v = data[tu.column_name].values

        if tu.variable_type == "factor_coded":
            _type = types.Factor
        else:
            _type = types.type_str(tu.variable_type)

        if _type.subtype_of(types.Discrete):
            v = _recode_discrete(v, tu.affected, tu.unaffected)

        if _type.subtype_of(types.Factor):
            try:
                code_name, v = _add_code_and_recode_factor(
                    v, mask, tu.database_name, tu.variable_type, tu.levels,
                    manager
                )
            except ValueError as e:
                message = "Unknown error"
                if len(e.args) == 1:
                    message = "{}: {}.".format(message, e.args[0])
                logger.warning(message)
                continue

        manager.add_phenotype(
            name=tu.database_name,
            code_name=code_name,
            snomed=tu.snomed_ct,
            parent=tu.parent,
            variable_type=_type.__name__,
            description=tu.description,
        )

        try:
            manager.add_data(tu.database_name, v)
        except types.InvalidValues as e:
            manager.delete(tu.database_name, _db_only=True)
            logger.warning(
                "{} It was be skipped.".format(e.message)
            )


def _recode_discrete(v, affected, unaffected):
    out = np.full(v.shape[0], np.nan, dtype=float)

    out[v == affected] = 1
    out[v == unaffected] = 0

    return out


def _add_code_and_recode_factor(v, mask, name, type_name, levels, manager):
    code = {}

    # Validate the levels.
    try:
        levels = json.loads(levels)
    except Exception:
        logger.warning(
            "Variable '{}' will not be imported because the levels field did "
            "not contain valid JSON.".format(name)
        )
        raise ValueError()

    # Check that all data values are represented.
    if not _have_coherent_mapping(v[mask], levels.keys()):
        logger.info(
            "Some data values for '{}' have no mappings in the levels field. "
            "Such values will be set to missing.".format(name)
        )

    if len(levels.values()) != len(set(levels.values())):
        logger.warning(
            "Variable '{}' will not be imported because the values in "
            "the levels field contains duplicate values. Did you assign "
            "labels in the configuration file?".format(name)
        )
        raise ValueError()

    if type_name == "factor":
        # Recode v.
        recoded = np.full(v.shape[0], np.nan, dtype=float)
        for s, i in levels.items():
            try:
                i = int(i)
            except TypeError:
                logger.warning(
                    "Variable '{}' will not be imported because the levels "
                    "field contains non-numeric code values."
                    "".format(name)
                )
            recoded[v == s] = i
            code[i] = s

    elif type_name == "factor_coded":
        for i, s in levels.items():
            i = int(i)
            code[i] = s

        # Ignore the values which are not in the code.
        for i in range(v.shape[0]):
            if mask[i] and (v[i] not in code.keys()):
                v[i] = np.nan

        recoded = v  # No need to further recode.

    # Check if code exists.
    code_name = None
    codes = manager.get_code_names()
    for _code_name in codes:
        db_code = dict(manager.get_code(_code_name))
        if code == db_code:
            # Found a valid code.
            code_name = _code_name
            break

    # Add code.
    if code_name is None:
        code_name = "_{}_code".format(name)
        manager.add_codes([(code_name, k, v) for k, v in code.items()])
        manager.con.commit()

    return code_name, recoded


def _have_coherent_mapping(values, keys):
    # Regular factors don't need type conversions.
    if type(values[0]) is str:
        return set(values) == set(keys)

    # Coded factors need integer comparisons.
    return {int(i) for i in values} == {int(i) for i in keys}


def _quote_li(li):
    return ["'{}'".format(str(s)) for s in li]
