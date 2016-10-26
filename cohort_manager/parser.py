"""
Parse the Excel file used to build the cohort.
"""

import logging
import json

import pandas as pd
import numpy as np

from .core import CohortManager
# from . import inference
# from . import types


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
    dups = dups ^ (variables["variable_type"] == "unique_key")

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
        usecols=list(variables["column_name"])
    )

    key_name = variables.iloc[key, :]["column_name"]

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
    for tu in variables.itertuples():
        if tu.variable_type == "unique_key":
            continue

        manager.add_phenotype(
            name=tu.database_name,
            snomed=tu.snomed_ct,
            parent=tu.parent,
            variable_type=tu.variable_type,
            description=tu.description,
        )

        manager.add_data(tu.database_name, data[tu.column_name])


def _quote_li(li):
    return ["'{}'".format(str(s)) for s in li]
