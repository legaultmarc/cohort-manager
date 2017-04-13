#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
A helper script to parse freetext pharmacotherapy data.
"""

import datetime
import collections
import argparse
import logging
import os

import pandas as pd
import numpy as np

from ..drugs import drug_search as ds
from ..core import CohortManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CURATE_DB_FILENAME = "drug_database_for_curation.xlsx"


def _read_drugs_file(args):
    logger.info("Reading pharmacotherapy file '{}'.".format(args.filename))
    logger.info("Columns:")
    dtypes = {}
    converters = {}

    def _date_parser(x):
        if x:
            try:
                return datetime.datetime.strptime(x, args.date_format).date()
            except ValueError:
                pass

        return None

    col_args = (
        ("Drug", getattr(args, "drug_column"), str),
        ("Sample", getattr(args, "sample_column"), str),
        ("Start date", getattr(args, "start_date_column", None), "date"),
        ("End date", getattr(args, "end_date_column", None), "date"),
        ("Indication", getattr(args, "indication_column", None), str),
        ("Dose", getattr(args, "dose_column", None), float),
    )
    for name, col, dtype in col_args:
        if col is not None:
            logger.info("\t- {} ({})".format(name, col))
            dtypes[col] = dtype if dtype != "date" else str

            if dtype == "date":
                converters[col] = _date_parser

    # Open the pharmacotherapy file.
    cols = list(dtypes.keys())

    params = {
        "dtype": dtypes,
        "usecols": cols,
        "converters": converters
    }

    if args.delim:
        params["sep"] = args.delim

    drugs_data = pd.read_csv(args.filename, **params)

    # Check that the required columns are in the file.
    if args.sample_column not in drugs_data.columns:
        raise ValueError("Sample column '{}' could not be found in input file."
                         "".format(args.sample_column))

    if args.drug_column not in drugs_data.columns:
        raise ValueError("Drug column '{}' could not be found in input file."
                         "".format(args.drug_column))

    logger.info("Read {} rows.".format(drugs_data.shape[0]))
    return drugs_data


def create_curation_file(args):
    drugs_data = _read_drugs_file(args)

    # If there is a custom database, we need to read it.
    custom = None
    if args.custom_database:
        custom = pd.read_csv(args.custom_database)
        custom = custom[["name", "molregno"]]

    # Create the drug correspondance table.
    _parse_freetext_drugs(drugs_data[args.drug_column].astype(str),
                          min_score=args.similarity_score_threshold,
                          custom=custom)

    command = ("cohort-drugs-import build "
               "--delim '{delim}' "
               "--sample-column '{sample_column}' "
               "--drug-column '{drug_column}' "
               "--import-file drug_database_for_curation.xlsx "
               "--cohort-name MY_COHORT '{filename}'")

    command = command.format(
        sample_column=args.sample_column,
        drug_column=args.drug_column,
        filename=args.filename,
        delim=args.delim if args.delim else ","
    )

    print(
        "\nA file containing matches for all the queries has been generated.\n"
        "Please open '{}' using a text editor or a spreadsheet editor to:\n\n"
        "1. Manually fill queries that were not matched automatically.\n"
        "2. Curate automatically matched drugs.\n"
        "To help you with this task, the REPL contains useful commands to "
        "search and get information on ChEMBL ('drug-search' and 'drug-info')."
        "\nWhen this process is complete, run the following command:\n\n"
        "{}\n\n".format(CURATE_DB_FILENAME, command)
    )


def _parse_freetext_drugs(names, min_score, custom):
    """Parse a list of freetext drug names and create a delimited file to allow
    manual curation.

    """
    if custom is not None:
        logger.info("Adding the custom database with {} entries to the "
                    "matching algorithm.".format(custom.shape[0]))
        ds.add_custom_database(custom)

    logger.info("Minimum similarity score is {}".format(min_score))
    logger.info("Matching freetext to ChEMBL (this might take a while).")
    matching_drugs = ds.find_drugs_in_queries(names, min_score)

    # Fix hierarchy.
    logger.info("Keeping the parent drug when necessary (multiple matching "
                "molregno).")
    matching_drugs = ds.fix_hierarchical_matches(names, matching_drugs)

    # Write the file to disk and show the user for manual curation.
    ds.write_results(CURATE_DB_FILENAME, names, matching_drugs)


def build_database(args):
    print("A cohort-manager drug database will be built using the provided "
          "file.")

    drugs_data = _read_drugs_file(args)

    # Read the curated database.
    logger.info(
        "Reading the curated drug database file '{}'.".format(args.import_file)
    )
    curated = pd.read_excel(
        args.import_file, sheetname=None
    )
    sheets = list(curated.keys())
    for sheet in sheets:
        try:
            curated[sheet] = curated[sheet][["query", "molregno"]]
        except:
            logger.warning(
                "Could not find columns 'query' and/or 'molregno' in sheet "
                "'{}'.".format(sheet)
            )
            del curated[sheet]

    curated = pd.concat(curated.values())

    curated.dropna(how="any", inplace=True)

    # Get the cohort.
    path, cohort_name = os.path.split(args.cohort_name)
    cohort = CohortManager(cohort_name, path=path)

    # Query to list of molregnos.
    query_dict = collections.defaultdict(list)
    for i, row in curated.iterrows():
        query_dict[row.query].append(int(row.molregno))

    if cohort.get_samples() is None:
        # This sample order has not been set yet.
        cohort.set_samples(
            drugs_data.loc[:, args.sample_column].unique()
        )

    # Load the pharmacotherapy data.
    no_matches = set()
    multiple_matches = set()
    molregnos = set()
    samples = set()
    n_entries = 0

    logger.info("Inserting pharmacotherapy data in the database (this might "
                "take a while).")
    for i, row in drugs_data.iterrows():
        sample = row[args.sample_column]
        drug_query = row[args.drug_column]

        # Optional parameters.
        optionals = {}
        date_opts = [
            ("start_date_column", "from_date"),
            ("end_date_column", "end_date"),
        ]

        for col, option in date_opts:
            if getattr(args, col, False):
                val = row[getattr(args, col)]
                optionals[option] = val

                if val and type(val) is not datetime.date:
                    logger.critical(
                        "Some entries in '{}' could not be parsed ('{}'). "
                        "Verify the dataset and the --date-format argument."
                        "".format(col, val)
                    )
                    quit(1)

        if getattr(args, "end_date_column", False):
            optionals["end_date"] = row[args.end_date_column]

        if getattr(args, "indication_column", False):
            optionals["indication"] = row[args.indication_column]

        if getattr(args, "dose_column", False):
            optionals["dose"] = row[args.dose_column]

        if drug_query not in query_dict:
            no_matches.add(drug_query)
            continue

        if len(query_dict[drug_query]) > 1:
            multiple_matches.add(drug_query)

        for molregno in query_dict[drug_query]:
            n_entries += 1
            molregnos.add(molregno)
            samples.add(sample)
            cohort.register_drug_user(molregno, sample, **optionals)

    cohort.con.commit()

    info = (n_entries, len(molregnos), len(samples), len(no_matches),
            len(multiple_matches))

    print(
        "\nSuccessfully built the pharmacotherapy database.\n"
        "A total of {} entries were recorded on {} different medications for "
        "{} samples.\n"
        "{} drugs had no matches in the curated file and have been ignored.\n"
        "{} drugs had multiple matches in the curated file and samples have "
        "been marked as users of all the corresponding drugs.".format(*info)
    )


def parse_args():
    description = (
        "Helper script to parse drug names from pharmacotherapy data.\n\n"
        "A delimited file in a long format (sample, drug) is used for input. "
        "The drugs freetext strings will be used to query ChEMBL and to "
        "normalize the drug identification using ChEMBL.\n"
        "A tab-delimited file will be created to allow the user to manually "
        "curate inference and correct any problems. This file can then be "
        "re-input to this script to build the full pharmacotherapy database "
        "containing information on pharmacotherapy for all the samples.\n"
        "This completed database is integrated with the cohort manager and "
        "can be viewed as a special phenotype."
    )

    parser = argparse.ArgumentParser(description=description)

    p_parser = argparse.ArgumentParser(add_help=False)

    group = p_parser.add_argument_group("Pharmacotherapy file options")
    group.add_argument(
        "--sample-column",
        help="Column of the file that contains sample IDs.",
        type=str,
        required=True,
    )

    group.add_argument(
        "--drug-column",
        help="Column of the file that contains the drugs.",
        type=str,
        required=True,
    )

    group.add_argument(
        "--delim", "-d",
        help="Delimiter character.",
        type=str,
    )

    group.add_argument(
        "filename",
        help="File containing the pharmacotherapy data.",
        type=str,
    )

    # Subparsers.
    subparser = parser.add_subparsers(
        dest="command",
        help=("Either parse the pharmacotherapy file or build the cohort "
              "manager database.")
    )
    subparser.required = True

    # Parse the drug names from the pharmacotherapy file.
    parse_parser = subparser.add_parser(
        "parse",
        help="Parse pharmacotherapy data to produce the curation file.",
        parents=[p_parser],
    )

    parse_parser.add_argument(
        "--similarity-score-threshold",
        help=("Minimum similarity score threshold to identify matching drugs "
              "in ChEMBL"),
        type=float,
        default=ds.DEFAULT_MIN_SCORE,
    )

    parse_parser.add_argument(
        "--custom-database",
        help=("Custom CSV database of 'name' to 'molregno'. This really needs "
              "to be a valid CSV files as no delimiters can be provided."),
        default=None
    )

    # Build the database.
    build_parser = subparser.add_parser(
        "build",
        help="Add pharmacotherapy data to the cohort.",
        parents=[p_parser],
    )

    build_parser.add_argument(
        "--import-file",
        help=("An excel file translating queries to ChEMBL drugs. Usually, "
              "this file is automaticlaly generated using this script."),
        type=str,
        required=True,
    )

    build_parser.add_argument(
        "--cohort-name",
        help=("The name of the cohort to link the pharmacotherapy data to. "
              "This option is used when creating the database"),
        required=True,
    )

    # Pharmacotherapy metadata.
    build_parser.add_argument(
        "--indication-column",
        help=("The name of the column containing the textual description of "
              "indication for a given drug."),
        default=None
    )

    build_parser.add_argument(
        "--start-date-column",
        help=("The name of the column containing the start date of "
              "pharmacotherapy."),
        default=None
    )

    build_parser.add_argument(
        "--end-date-column",
        help=("The name of the column containing the start date of "
              "pharmacotherapy."),
        default=None
    )

    build_parser.add_argument(
        "--date-format",
        help=("A template string for the date representaiton. \nSee "
              "https://docs.python.org/3.5/library/datetime.html#strftime-strptime-behavior"
              "\nfor a list of available directives. The default is the "
              "ISO 8601 format (YYYY-MM-DD)."),
        default="%Y-%m-%d"
    )

    build_parser.add_argument(
        "--dose-column",
        help=("The name of the column containing the dose for drugs. This "
              "column should contain numbers."),
        default=None
    )

    # TODO
    # Add the --dose-unit-column and --dose-unit flags. This is not formally
    # available in most datasets so it is not a priority at this point.

    args = parser.parse_args()
    if args.command == "build":
        build_database(args)
    elif args.command == "parse":
        create_curation_file(args)


if __name__ == "__main__":
    parse_args()
