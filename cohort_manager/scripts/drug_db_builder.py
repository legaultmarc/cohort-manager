#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
A helper script to parse freetext pharmacotherapy data.
"""

import sqlite3
import collections
import argparse
import logging

import pandas as pd
import numpy as np

import cohort_manager.drugs.drug_search as ds
from cohort_manager.core import CohortManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CURATE_DB_FILENAME = "drug_database_for_curation.txt"


def _read_drugs_file(args):
    logger.info("Reading pharmacotherapy file '{}'.".format(args.filename))

    # Open the pharmacotherapy file.
    if args.delim:
        drugs_data = pd.read_csv(args.filename, sep=args.delim)
    else:
        drugs_data = pd.read_csv(args.filename)

    # Check that the required columns are in the file.
    if args.sample_column not in drugs_data.columns:
        raise ValueError("Sample column '{}' could not be found in input file."
                         "".format(args.sample_column))

    if args.drug_column not in drugs_data.columns:
        raise ValueError("Drug column '{}' could not be found in input file."
                         "".format(args.drug_column))

    drugs_data = drugs_data[[args.sample_column, args.drug_column]]
    logger.info("Read {} rows.".format(drugs_data.shape[0]))
    return drugs_data


def create_curation_file(args):
    drugs_data = _read_drugs_file(args)

    # Create the drug correspondance table.
    _parse_freetext_drugs(drugs_data[args.drug_column])

    print(
        "\nA file containing matches for all the queries has been generated.\n"
        "Please open '{}' using a text editor or a spreadsheet editor to:\n\n"
        "1. Curate automatically matched drugs.\n"
        "2. Manually fill queries that were not matched automaticall.\n"
        "To help you with this task, the REPL contains useful commands to "
        "search and get information on ChEMBL ('drug-search' and 'drug-info')."
        "\nWhen this process is complete, run the following command:\n\n"
        "drug-db-builder --database drug_database_for_curation.txt --cohort "
        "COHORT_NAME pharmacotherapy_file.txt\n\n".format(CURATE_DB_FILENAME)
    )


def _parse_freetext_drugs(names):
    """Parse a list of freetext drug names and create a delimited file to allow
    manual curation.

    """
    logger.info("Matching freetext to ChEMBL (this might take a while).")
    matching_drugs = ds.find_drugs_in_queries(names)

    # Remove small substring matches.
    logger.info("Removing short substring matches.")
    matching_drugs = ds.remove_substring_matches(names, matching_drugs)

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
        "Reading the curated drug database file '{}'.".format(args.database)
    )
    curated = pd.read_csv(args.database, sep=",", header=0)
    curated = curated[["query", "molregno"]]

    # Get the cohort.
    cohort = CohortManager(args.cohort)

    # Query to list of molregnos.
    query_dict = collections.defaultdict(list)
    for i, row in curated.iterrows():
        if not np.isnan(row.molregno):
            query_dict[row.query].append(int(row.molregno))

    # Load the pharmacotherapy data.
    no_matches = set()
    multiple_matches = set()
    molregnos = set()
    samples = set()
    n_entries = 0
    for i, row in drugs_data.iterrows():
        sample = row[args.sample_column]
        drug_query = row[args.drug_column]

        if drug_query not in query_dict:
            no_matches.add(drug_query)
            continue

        if len(query_dict[drug_query]) > 1:
            multiple_matches.add(drug_query)

        for molregno in query_dict[drug_query]:
            n_entries += 1
            molregnos.add(molregno)
            samples.add(sample)
            try:
                cohort.register_drug_user(molregno, sample)
            except sqlite3.IntegrityError:
                pass  # It was already in the database (could be duplicates).

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

    parser.add_argument(
        "filename",
        help="File containing the pharmacotherapy data.",
        type=str,
    )

    parser.add_argument(
        "--sample-column",
        help="Column of the file that contains sample IDs.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--drug-column",
        help="Column of the file that contains the drugs.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--delim", "-d",
        help="Delimiter character.",
        type=str,
    )

    parser.add_argument(
        "--database",
        help=("A CSV file translating queries to ChEMBL drugs. Usually, this "
              "file is automaticlaly generated using this script."),
        type=str,
        default=None,
    )

    parser.add_argument(
        "--cohort",
        help=("The name of the cohort to link the pharmacotherapy data to. "
              "This option is used when creating the database"),
        default=None
    )

    parser.add_argument(
        "--similarity-score-threshold",
        help=("Minimum similarity score threshold to identify matching drugs "
              "in ChEMBL"),
        type=str,
        default=None,
    )

    args = parser.parse_args()
    if args.database and args.cohort:
        build_database(parser.parse_args())
    else:
        if args.database or args.cohort:
            print("The --database and --cohort options need to be used "
                  "together when ready to build the pharmacotherapy database "
                  "for a given cohort.")
            quit(1)
        create_curation_file(parser.parse_args())


if __name__ == "__main__":
    parse_args()
