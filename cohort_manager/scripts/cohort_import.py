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


def create_import_file(filenames, delimiter, encoding, known_missings):
    """Parse data files and create the import file."""

    frames = []
    for fn in filenames:
        with codecs.open(fn, "r", encoding) as f:
            path = os.path.abspath(fn)
            df = _parse_file(f, path, delimiter, encoding, known_missings)
            frames.append(df)

    dataset = pd.concat(frames)

    dataset_filename = "cohort_manager_import.xlsx"

    writer = pd.ExcelWriter(dataset_filename, engine="xlsxwriter")
    dataset.to_excel(writer, "Variables", index=False)

    workbook = writer.book
    meta_sheet = workbook.add_worksheet("Metadata")

    meta_sheet.write_string(0, 0, "key")
    meta_sheet.write_string(0, 1, "value")

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


def _parse_file(f, path, delimiter, encoding, known_missings):
    examples = collections.defaultdict(list)

    reader = csv.reader(f, delimiter=delimiter)
    names = next(reader)

    # Read up to 7000 examples.
    i = 0
    while i < 7000:
        try:
            row = next(reader)
        except StopIteration:
            break

        for j, value in enumerate(row):
            examples[names[j]].append(value)

    # Type cast the dataset.
    dataset = []
    for column in examples:
        type_name = inference.infer_type(examples[column],
                                         known_missings=known_missings)
        t = None
        try:
            t = types.type_str(type_name)
        except Exception:  # Type is not supported.
            pass

        # Write the Excel file and ask the user for curation.
        if type_name is None:
            type_name = ""

        levels = ""
        affected = unaffected = ""
        if t is not None and t.subtype_of(types.Discrete):
            try:
                code, _ = inference.cast_type(
                    [i for i in examples[column] if i not in known_missings],
                    t
                )
                affected = code.get(1, "")
                unaffected = code.get(0, "")
            except ValueError:
                # Fallback to factor coded.
                type_name = "factor_coded"

        # Need to start another if because type_name could have changed.
        if type_name == "factor":
            levels = json.dumps(inference.infer_levels(examples[column]))

        elif type_name == "factor_coded":
            levels = json.dumps({
                level: level for level in set(examples[column])
                if level not in known_missings
            })

        import_flag = type_name not in ("", "text")

        description = snomed = parent = ""

        dataset.append((column, column, path, parent, type_name, affected,
                        unaffected, levels, description, snomed, import_flag))

    dataset = pd.DataFrame(
        dataset,
        columns=("column_name", "database_name", "path", "parent",
                 "variable_type", "affected", "unaffected", "levels",
                 "description", "snomed_ct", "import")
    )

    return dataset


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
        "filenames",
        help="Filename(s) of the delimited files containing phenotype data.",
        type=str,
        nargs="+",
        action="store"
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
        help="Known missing values (separated by spaces).",
        nargs="*",
        default=["NA"]
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
        type=str,
        required=True
    )

    build_parser.add_argument(
        "--cohort-name",
        help="The name or path to the cohort.",
        required=True
    )

    args = parser.parse_args()

    if args.command == "parse":
        known_missings = [""]
        known_missings += [s.lstrip("\\") for s in args.known_missings]

        create_import_file(
            args.filenames, args.delimiter, args.encoding, known_missings,
        )
    elif args.command == "build":
        success = cm_parser.import_file(args.import_file, args.cohort_name)
        if not success:
            quit(1)


if __name__ == "__main__":
    parse_args()
