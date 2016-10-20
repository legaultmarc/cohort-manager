"""
Parse the Excel file used to build the cohort.
"""

import logging
import collections
import json

import pandas as pd
import numpy as np

from .core import CohortManager
from . import inference
from . import types


logger = logging.getLogger(__name__)


class InvalidConfig(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Invalid configuration file.\n{}".format(self.value)


def import_file(filename, cohort_name):
    """Fill the database using the provided Excel configuration file."""
    # Read the columns data.
    variables = pd.read_excel(filename, sheetname="Variables")

    # Keep only variables that are flagged for importation.
    variables = variables.loc[variables["import"].astype(int) == 1, :]

    # Set the name as the index.
    try:
        variables = variables.set_index("name", verify_integrity=True)
    except ValueError:
        dups = list(variables.loc[variables.duplicated("name"), "name"])
        logger.critical(
            "The import file contains duplicate names ({}). "
            "Please make sure that entries from the 'name' column are unique."
            "".format(dups)
        )
        return

    # Read the metadata sheet containing information about the source file.
    metadata = pd.read_excel(filename, sheetname="Metadata", header=None)

    # Default values.
    metadata_dict = {"encoding": "utf-8", "delimiter": ",",
                     "known_missings": "[]"}
    for i, row in metadata.iterrows():
        metadata_dict[row[0]] = row[1]

    metadata_dict["known_missings"] = json.loads(
        metadata_dict["known_missings"]
    )

    manager = CohortManager(cohort_name)
    for filename in variables["path"].unique():
        parse_file(manager, filename, variables, metadata_dict)


def get_sample_id_column(filename, variables):
    keys = variables.loc[variables["variable_type"] == "unique_key", :].index
    keys = list(keys)

    if len(keys) == 0:
        logger.critical(
            "Can't import phenotypes from '{}' because there is no field of "
            "type 'unique_key' (for the sample IDs).".format(filename)
        )
        return False

    elif len(keys) > 1:
        logger.critical(
            "Can't import phenotypes from '{}' because the column containing "
            "sample IDs is not unique (only one column of type 'unique_key' "
            "should be in the import file.".format(filename)
        )
        return False

    return keys.pop()


def parse_file(manager, filename, variables, metadata):
    logger.info("Processing data from '{}'.".format(filename))

    # Use only the relevant subset.
    variables = variables.loc[variables["path"] == filename, :]

    # Make sure names are unique within this subset.
    if variables["col_name"].unique().shape[0] != variables.shape[0]:
        logger.critical(
            "Column name definitions are not unique for file '{}'."
            "".format(filename)
        )
        return

    # Get the name of the sample ID column.
    sample_column = get_sample_id_column(filename, variables)
    if not sample_column:
        return

    samples = []

    sep = metadata["delimiter"]
    with open(filename, "r", encoding=metadata["encoding"]) as f:
        header = next(f).strip()
        header = header.split(sep)

        variable_to_index = dict([(j, i) for i, j in enumerate(header)])
        data = collections.defaultdict(list)

        for line in f:
            line = line.rstrip().split(sep)

            # Build the list of sample IDs at the same time.
            samples.append(line[variable_to_index[sample_column]])

            # Keep only the data.
            for variable in variables["col_name"]:
                if variable == sample_column:
                    continue

                try:
                    index = variable_to_index[variable]
                except KeyError:
                    logger.critical(
                        "Could not find column '{}' in file '{}'."
                        "".format(variable, filename)
                    )

                if line[index] in metadata["known_missings"]:
                    data[variable].append("")
                else:
                    data[variable].append(line[index])

    # Type cast and insert.
    for variable in data.keys():
        row = int(np.where(variables["col_name"] == variable)[0][0])
        try:
            _type = types.type_str(variables.iloc[row]["variable_type"])
        except Exception:
            logger.warning("Could not load variable '{}' because of it has an "
                           "unknown type.".format(variable))
            continue

        if _type.subtype_of(types.Discrete):
            # Encode manually.
            # Get the code.
            affected = float(variables.iloc[row]["affected"])
            unaffected = float(variables.iloc[row]["unaffected"])

            v = np.empty(len(data[variable]), dtype=np.float)
            for i in range(v.shape[0]):
                value = data[variable][i]

                try:
                    value = float(value)
                except ValueError:
                    v[i] = np.nan
                    continue

                if value == affected:
                    v[i] = 1
                elif value == unaffected:
                    v[i] = 0
                else:
                    v[i] = np.nan

            data[variable] = v

        # We don't do recoding for non-discrete variables.
        else:
            try:
                data[variable] = inference.cast_type(data[variable], _type)[1]
            except ValueError:
                logger.warning(
                    "Could not insert variable '{}' because its data type "
                    "({}) can't be automatically encoded."
                    "".format(variable, _type.__name__)
                )
                continue

        # Check if it is a new cohort and set or check the sample order.
        if not (manager["frozen"] == "yes"):
            manager.set_samples(samples)
        else:
            samples = np.array(samples)
            db_samples = manager.get_samples()
            if not np.all(samples == db_samples):
                raise NotImplementedError(
                    "Automatic import of permutated samples is not yet "
                    "supported."
                )

        # Get other meta-data.
        description = variables.iloc[row]["description"]
        snomed = variables.iloc[row]["snomed-ct"]

        name = variables.index[row]

        manager.add_phenotype(
            name=name, variable_type=_type.__name__,
            description=description, snomed=snomed,
        )

        try:
            manager.add_data(
                name, data[variable]
            )
        except types.InvalidValues as e:
            logger.warning(e.message)
            manager.delete(variable, _db_only=True)

    # Build hierarchy.
    _build_hierarchy(variables["parent"].dropna(), manager)

    manager.commit()
    return True


def _build_hierarchy(hierarchy, manager):
    """Update phenotype parent pointers."""
    available_phenotypes = set(manager.get_phenotypes_list())
    for child, parent in hierarchy.iteritems():
        # Check that the parent exists.
        if parent not in available_phenotypes:
            logger.warning(
                "Could not set parent for '{}' to '{}' because the parent is "
                "not in the manager.".format(child, parent)
            )
            continue

        manager.update_phenotype(child, parent=parent)

    manager.commit()
