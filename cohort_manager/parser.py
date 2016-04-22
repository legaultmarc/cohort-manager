"""
Parse the YAML file used to build the cohort.
"""

import os
import shutil
import logging

import yaml
import pandas as pd
import numpy as np

from .core import CohortManager, UnknownSamplesError, vector_map


ENGINES = {}
logger = logging.getLogger(__name__)


class InvalidConfig(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Invalid configuration file.\n{}".format(self.value)


def parse_yaml(filename):
    with open(filename) as f:
        config = yaml.load(f)

    # name
    name = config.get("name")
    if not name:
        raise InvalidConfig("No cohort name ('name') was provided.")

    # FIXME. In production, be more careful.
    if os.path.isdir(name):
        logger.warning("Deleting directory '{}'.".format(name))
        shutil.rmtree(name)

    manager = CohortManager(name)

    # samples
    samples = config.get("samples")
    if samples:
        manager.set_samples(_parse_list_from_file(samples))

    # codes
    codes = config.get("codes")
    if not codes:
        logger.warning("No codes were defined for the cohort. All variables "
                       "of type 'factor' should have defined codes.")
    for code in codes:
        _handle_code_node(manager, code)

    # dummy phenotypes
    dummy_phenotypes = config.get("dummies", [])
    for dummy in dummy_phenotypes:
        manager.add_dummy_phenotype(dummy)

    # data
    data = config.get("data", [])

    for elem in data:
        _handle_file_node(manager, elem)

    manager.commit()

    # If there is a recodeAsFactor operation (to merge multiple discrete
    # variables to a single factor variable), it will be done here because
    # the database has been filled.
    recode_as_factors = config.get("recodeAsFactors", [])

    for elem in recode_as_factors:
        _handle_recode_as_factors(manager, elem)

    # Create virtual variables.
    virtuals = config.get("virtuals", [])

    for elem in virtuals:
        _handle_virtuals(manager, elem)

    return manager


def _parse_list_from_file(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


def _handle_code_node(manager, code):
    name = code.get("name")
    if not name:
        raise InvalidConfig("Code nodes need a name.")

    code = code.get("code")
    if type(code) is not list:
        raise InvalidConfig("A 'code' needs to be specified as a list.")

    # Check code name uniqueness.
    if name in manager.get_code_names():
        raise InvalidConfig("Code '{}' already exists.".format(name))

    # Check code validity (int -> str)
    for i in range(len(code)):
        try:
            code[i][0] = int(code[i][0])
        except ValueError:
            raise InvalidConfig(
                "Code keys need to be integers. Got '{}'.".format(code[i][0])
            )
        code[i][1] = str(code[i][1])

        manager.add_code(name, code[i][0], code[i][1])

    manager.commit()


def _handle_recode_as_factors(manager, node):
    new_name = node.get("name")
    if new_name is None:
        raise InvalidConfig("recodeAsFactors mappings need a 'name' field.")

    delete = node.get("delete", False)
    if type(delete) is not bool:
        raise InvalidConfig("recodeAsFactors mappings' delete field requires "
                            "a bool.")

    # list of phenotypes to merge.
    phenotypes = node.get("phenotypes")
    data = np.zeros(manager.n) - 1  # Initialize at -1.
    for code_int, phenotype in enumerate(phenotypes):
        # Check that it's a discrete variable.
        meta = manager.get_phenotype(phenotype)
        if meta["variable_type"] != "discrete":
            raise InvalidConfig("The recodeAsFactor function is only valid to "
                                "merge multiple discrete variables into a "
                                "single factor.")

        # Create the code.
        code_name = "{}Code".format(new_name)
        manager.add_code(code_name, code_int, phenotype)

        # Encode the data.
        data[manager.get_data(phenotype, numpy=True) == 1] = code_int

    # The remaining values (still -1) will be missing (nan).
    data[data == -1] = np.nan

    # Create the new phenotype and add the data.
    manager.add_phenotype(
        name=new_name,
        variable_type="factor",
        code_name=code_name
    )
    manager.commit()

    manager.add_data(new_name, data)

    # Delete the sub-phenotypes if necessary.
    if delete:
        for phenotype in phenotypes:
            manager.delete(phenotype)


def _handle_virtuals(manager, node):
    """Used to create virtual phenotypes at cohort creation.

    .. todo::

        This might become dead code.

    """
    name = node.pop("name", None)
    if name is None:
        raise InvalidConfig("No name provided for virtual variable.")

    _type = node.pop("type", None)
    if _type not in ("discrete", "continuous"):
        raise InvalidConfig("Invalid or missing type for virtual variable.")

    formula = node.pop("formula", None)
    if formula is None:
        raise InvalidConfig("Missing formula for virtual variable.")

    data = eval(formula, {}, {"v": manager.variable})

    # Create the new phenotype.
    manager.add_phenotype(name=name, variable_type=_type, **node)
    manager.add_data(name, data.data)
    manager.commit()


def _handle_file_node(manager, node):
    # Check filename validity.
    filename = node.get("filename")
    if not filename:
        raise InvalidConfig("No filename provided in data node.")
    elif not os.path.isfile(filename):
        raise InvalidConfig("Could not find '{}'.".format(filename))

    # Get the parser.
    engine = node.get("engine", "read_csv")
    if engine not in ENGINES:
        raise InvalidConfig("Unknown file parser '{}'.".format(engine))

    engine = ENGINES[engine]

    # Parse the file.
    data = engine(filename, node)

    # Set the sample order. The order of samples is shared for all variables
    # in a file, so we don't need to check at the variable parsing level.
    samples = manager.get_samples()
    if samples is not None:
        samples = np.array(samples)

        # Check that sample ids are consistent.
        extra = set(data.index.values) - set(samples)
        if extra:
            msg = ("Samples that were not in the initial list have been added "
                   "('{}').").format(", ".join(list(extra)))

            raise UnknownSamplesError(msg)

        data = data.loc[samples, :]
        assert np.all(samples == data.index)
    else:
        manager.set_samples(data.index.values.astype(np.string_))

    # Add information on all variables.
    for variable in node.get("variables", []):
        _handle_variable_node(manager, variable, data)


def _handle_variable_node(manager, node, data):
    column = node.get("column")
    if not column:
        raise InvalidConfig("Variable node has no 'column'.")

    # Parse the variable node attributes.
    empty_are_controls = node.get("emptyAreControls", False)
    name = node.get("name", column)
    icd10 = node.get("icd10")
    parent = node.get("parent")
    _map = node.get("map")
    _type = node.get("type")
    code = node.get("code")
    crf_page = node.get("crf_page")
    if crf_page:
        crf_page = int(crf_page)
    description = node.get("description")

    # For factors, a code is required.
    if _type == "factor" and code is None:
        raise InvalidConfig("A 'code' needs to be provided for categorical "
                            "variables ('type: factor').")

    # Extract the data from the file.
    try:
        data = data.loc[:, column].values.copy()
    except KeyError:
        raise InvalidConfig(
            "Could not extract data from column '{}'.".format(column)
        )

    # The user can use emptyAreControls: True so that empty (normally missing)
    # values are understood as controls.
    if empty_are_controls:
        data[np.isnan(data)] = 0

    # Apply mapper.
    if _map:
        for i, tu in enumerate(_map):
            if tu[1] == "nan":
                _map[i][1] = np.nan

            elif tu[0] == 0:
                if empty_are_controls and _type == "discrete":
                    logger.warning("When using emptyAreControls, remapping "
                                   "controls (0) will also affect empty "
                                   "fields.")

            # We always cast the target to float.
            else:
                _map[i][1] = float(_map[i][1])

        if _type == "factor":
            # We can only add NaNs when remapping factors.
            for tu in _map:
                if not np.isnan(tu[1]):
                    raise TypeError("Factor remapping only allows remapping "
                                    "to NaN.")
        elif _type not in ("continuous", "discrete"):
            raise TypeError("Unknown variable type ('{}') in variable node."
                            "".format(_type))

        data = vector_map(data, _map)

    # Add the phenotype to the database.
    manager.add_phenotype(
        name=name,
        icd10=icd10,
        parent=parent,
        variable_type=_type,
        crf_page=crf_page,
        description=description,
        code_name=code
    )

    manager.add_data(name, data)


class engine(object):
    def __init__(self, key):
        """Decorator to register file parsing engines."""
        self.key = key

    def __call__(self, f):
        global ENGINES
        ENGINES[self.key] = f
        return f


@engine("read_csv")
def _csv_engine(filename, node):
    """CSV parser based on pandas.

    It is assumed that the returned value for parsers is compatible with
    pandas indexing of a samples x phenotype data frame.

    """
    sep = node.get("sep", ",")
    header = node.get("header", 0)
    logger.debug(
        "Parsing CSV '{}'. sep={}, header={}.".format(filename, sep, header)
    )
    index = node.get("index")
    encoding = node.get("encoding")
    if not index:
        raise InvalidConfig("An 'index' column is required. It should "
                            "be the sample id column.")

    df = pd.read_csv(filename, sep=sep, header=header, encoding=encoding)
    df.set_index(index, verify_integrity=True, inplace=True, drop=True)
    df.index = df.index.astype(str)

    return df
