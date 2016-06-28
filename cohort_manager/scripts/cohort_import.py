#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
Helper script to build a cohort from data sources.
"""

import collections
import argparse
import logging
import codecs
import json
import csv
import os

import pandas as pd

from .. import inference
from .. import types
from .. import parser as cm_parser


logger = logging.getLogger(__name__)


def create_skeleton_from_file(filename, delimiter, encoding, known_missings):
    """Infer data type from a filename."""
    inferred_types = {}
    examples = collections.defaultdict(list)

    with codecs.open(filename, "r", encoding) as f:
        reader = csv.reader(f, delimiter=delimiter)
        names = next(reader)

        # Read up to 5000 examples.
        i = 0
        while i < 5000:
            try:
                row = next(reader)
            except StopIteration:
                break

            for j, value in enumerate(row):
                if value in known_missings:
                    examples[names[j]].append("")
                else:
                    examples[names[j]].append(value)
            i += 1

    # Infer types from examples.
    for column in examples:
        inferred_types[column] = inference.infer_type(examples[column])

    # Type cast the dataset.
    codes = {}
    for column in examples.keys():
        t = None
        try:
            t = types.type_str(inferred_types[column])
        except Exception:  # Type is not supported.
            pass

        code, examples[column] = inference.cast_type(examples[column], t)
        codes[column] = code

    # Infer relationship between columns.
    mat = inference.build_mi_matrix(examples, inferred_types)
    names, _ = mat

    clusters = inference.hierarchical_clustering(mat, examples, inferred_types,
                                                 codes)
    name_cluster = list(zip(names, clusters))
    cluster_numbers = collections.Counter(clusters)

    # Write the Excel file and ask the user for curation.
    # Rows: (group, name, parent, variable_type, import?)
    dataset = []
    for n in cluster_numbers:
        for col, cluster in name_cluster:
            if cluster == n:
                type_name = inferred_types[col]
                if type_name is None:
                    type_name = ""

                affected = unaffected = ""
                code = codes[col]
                if code is not None:
                    affected = code.get(1, "")
                    unaffected = code.get(0, "")

                import_flag = bool(type_name and type_name != "text")

                description = icd10 = ""

                dataset.append((n, col, "", type_name, affected, unaffected,
                                description, icd10, import_flag))

    dataset = pd.DataFrame(
        dataset,
        columns=("group", "name", "parent", "variable_type", "affected",
                 "unaffected", "description", "icd10", "import")
    )

    dataset_filename = os.path.basename(
        os.path.splitext(filename)[0]
    ) + "_import.xlsx"

    writer = pd.ExcelWriter(dataset_filename, engine="xlsxwriter")
    dataset.to_excel(writer, "Variables", index=False)

    workbook = writer.book
    meta_sheet = workbook.add_worksheet("Metadata")

    meta_sheet.write_string(0, 0, "filename")
    meta_sheet.write_string(0, 1, filename)

    meta_sheet.write_string(1, 0, "delimiter")
    meta_sheet.write_string(1, 1, delimiter)

    meta_sheet.write_string(2, 0, "encoding")
    meta_sheet.write_string(2, 1, encoding)

    meta_sheet.write_string(3, 0, "known_missings")
    meta_sheet.write_string(3, 1, json.dumps(known_missings))

    writer.save()

    print("\nA file ({filename}) containing all the information necessary to "
          "import the provided dataset into CohortManager has been generated."
          "\nVerify that all the types have been inferred correctly and use "
          "the 'build' sub-command of this script to fill the database.\n\n"
          "A sample command would be:\n\n"
          "cohort-import build --import-file '{filename}' "
          "--cohort-name MY_COHORT\n\n"
          "".format(filename=dataset_filename))


def parse_args():
    description = (
        "Helper script to help import phenotype data into a cohort."
    )

    parser = argparse.ArgumentParser(description=description)

    parent_parser = argparse.ArgumentParser(add_help=False)

    subparser = parser.add_subparsers(
        dest="command",
        help=("Either parse a phenotype input file or build the cohort manager"
              "database.")
    )
    subparser.required = True

    parse_parser = subparser.add_parser(
        "parse",
        help="Parse phenotype input file.",
        parents=[parent_parser]
    )

    group = parse_parser.add_argument_group("Input file options")
    group.add_argument(
        "--filename",
        help="Filename of the CSV containing phenotype data.",
        type=str,
        required=True
    )

    group.add_argument(
        "--delimiter", "-d",
        help="Column delimiter.",
        default=","
    )

    group.add_argument(
        "--encoding",
        help="File encoding.",
        default="utf-8"
    )

    group.add_argument(
        "--known-missings",
        help="Known missing values (separated by comma).",
        nargs="*",
        default=["NA"]
    )

    group.add_argument(
        "--icd10-column",
        help="The name of the column containing ICD10 codes.",
        default=None
    )

    group.add_argument(
        "--description-column",
        help="The name of the column containing variable descriptions.",
        default=None
    )

    # Build Parser
    build_parser = subparser.add_parser(
        "build",
        help=("Fill the cohort manager database given the import description "
              "file."),
        parents=[parent_parser],
    )

    build_parser.add_argument(
        "--import-file",
        help=("The import description file generated during the 'parse' step. "
              "A list of import file names can also be provided to batch "
              "insert phenotype data (e.g. using bash globs)."),
        nargs="+",
        action="store"
    )

    build_parser.add_argument(
        "--cohort-name",
        help="The name or path to the cohort.",
        required=True
    )

    args = parser.parse_args()

    if args.command == "parse":
        known_missings = [""]
        known_missings += args.known_missings

        create_skeleton_from_file(
            args.filename, args.delimiter, args.encoding, known_missings,
            icd10_col=args.icd10_column,
            description_col=args.description_column
        )
    elif args.command == "build":
        at_least_one_failed = False
        for filename in args.import_file:
            success = cm_parser.parse_excel(filename, args.cohort_name)
            if not success:
                at_least_one_failed = True

        if at_least_one_failed:  # Return with non-zero.
            quit(1)


if __name__ == "__main__":
    parse_args()
