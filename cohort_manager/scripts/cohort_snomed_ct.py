#!/usr/bin/env python

"""
Helper script to install SNOMED-CT from the release format 2 (rf2) files.

This will build a PostgreSQL database containing a subset of the SNOMED-CT
files.

Installing SNOMED-CT is required to use the cohort_manager module.

"""

import os
import getpass
import argparse
import configparser
import collections
import csv
import logging

import psycopg2


logger = logging.getLogger(__name__)


def generate_configuration_file():
    config = configparser.ConfigParser()
    config["DATABASE"] = collections.OrderedDict([
        ("host", "127.0.0.1"),
        ("port", 5432),
        ("db_name", "snomed"),
        ("username", getpass.getuser()),
        ("password", "")
    ])

    refset = os.path.join("Full", "Refset")
    term = os.path.join("Full", "Terminology")

    # This is really long so I wrote it here to declutter.
    # The ICD-9 file is the ComplexMapFull.
    map_default_fn = "der2_iisssccRefset_ExtendedMapFull_INT_20160131.txt"

    config["SNOMED_CT_FILES"] = collections.OrderedDict([
        ("Refset_ExtendedMap", os.path.join(refset, "Map", map_default_fn)),
        ("Refset_Language", os.path.join(
            refset, "Language", "der2_cRefset_LanguageFull-en_INT_20160131.txt"
        )),
        ("Concept", os.path.join(term, "sct2_Concept_Full_INT_20160131.txt")),
        # The technical implementation guide suggests using the Relationship
        # format unless the provided software does Description logic inferences
        # from the stated representation (which we don't do).
        ("Relationship", os.path.join(
            term, "sct2_Relationship_Full_INT_20160131.txt"
        )),
        ("Description", os.path.join(
            term, "sct2_Description_Full-en_INT_20160131.txt"
        )),
    ])

    filename = "snomed_ct_config.ini"
    if not os.path.isfile(filename):
        print(
            "Writing the default SNOMED-CT configuration file to '{}'. \n"
            "This file was build using reasonable default values, but you \n "
            "will need to adapt it to your system's settings.".format(filename)
        )
        with open(filename, "w") as ini:
            config.write(ini)
    else:
        print("Could not create file '{}' because it already exists.\n"
              "Please delete this file manually if you wish to overwrite it "
              "with the\ndefault configuration values.".format(filename))


def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def build_database_from_config(config):
    config = parse_config(config)

    db_config = config["DATABASE"]
    db_params = dict(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["db_name"],
        user=db_config["username"],
        password=db_config["password"]
    )

    with psycopg2.connect(**db_params) as con:
        with con.cursor() as cur:
            cur.execute("CREATE SCHEMA snomed_ct")
            create_concept(cur)
            create_relationship(cur)
            create_description(cur)
            create_refset_language(cur)
            create_refset_extended_map(cur)

    # Read the files and insert values.
    sql = {}
    sql["description"] = (
        "INSERT INTO snomed_ct.Description "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )

    sql["relationship"] = (
        "INSERT INTO snomed_ct.Relationship "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )

    sql["concept"] = (
        "INSERT INTO snomed_ct.Concept "
        "VALUES (%s, %s, %s, %s, %s)"
    )

    sql["refset_language"] = (
        "INSERT INTO snomed_ct.Language "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )

    sql["refset_extendedmap"] = (
        "INSERT INTO snomed_ct.ExtendedMap "
        "VALUES ({})".format(", ".join(["%s" for i in range(13)]))
    )

    con = psycopg2.connect(**db_params)
    cur = con.cursor()

    filenames = config["SNOMED_CT_FILES"]

    for k in filenames:
        buf = []
        filename = filenames[k]

        if k not in sql:
            logger.warning(
                "Skipping data file '{}' (no database mapping)."
                "".format(filename)
            )
            continue

        logger.info(
            "Reading data from {} into the database.".format(filename)
        )

        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Header

            for row in reader:
                buf.append(tuple(row))

                if len(buf) > 100000:
                    cur.executemany(sql[k], buf)
                    con.commit()
                    buf = []

            if buf:
                cur.executemany(sql[k], buf)
                con.commit()
                buf = []

    cur.close()
    con.close()
    print("Successfully created the SNOMED-CT database.")


def create_concept(cur):
    logger.info("Creating table snomed_ct.Concept")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS snomed_ct.Concept ("
        "  id bigint NOT NULL,"
        "  effectiveTime integer NOT NULL,"
        "  active boolean NOT NULL,"
        "  moduleId bigint NOT NULL,"
        "  definitionStatusId bigint NOT NULL"
        ")"
    )
    cur.execute("CREATE INDEX idx_concept_id ON snomed_ct.Concept (id)")


def create_description(cur):
    logger.info("Creating table snomed_ct.Description")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS snomed_ct.Description ("
        "  id bigint NOT NULL,"
        "  effectiveTime bigint NOT NULL,"
        "  active boolean NOT NULL,"
        "  moduleId bigint NOT NULL,"
        "  conceptId bigint NOT NULL,"
        "  languageCode VARCHAR(10) NOT NULL,"
        "  typeId bigint NOT NULL,"
        "  term text NOT NULL,"
        "  caseSignificanceId bigint NOT NULL"
        ")"
    )
    cur.execute(
        "CREATE INDEX idx_desc_cpt ON snomed_ct.Description (conceptId)"
    )


def create_relationship(cur):
    logger.info("Creating table snomed_ct.Relationship")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS snomed_ct.Relationship ("
        "  id bigint NOT NULL,"
        "  effectiveTime integer NOT NULL,"
        "  active boolean NOT NULL,"
        "  moduleId bigint NOT NULL,"
        "  sourceId bigint NOT NULL,"
        "  destinationId bigint NOT NULL,"
        "  relationshipGroup integer NOT NULL,"
        "  typeId bigint NOT NULL,"
        "  characteristicTypeId bigint NOT NULL,"
        "  modifierId bigint NOT NULL"
        ")"
    )
    cur.execute(
        "CREATE INDEX idx_rel_src ON snomed_ct.Relationship (sourceId)"
    )
    cur.execute(
        "CREATE INDEX idx_rel_dst ON snomed_ct.Relationship (destinationId)"
    )
    cur.execute(
        "CREATE INDEX idx_rel_type ON snomed_ct.Relationship (typeId)"
    )


def create_refset_language(cur):
    logger.info("Creating table snomed_ct.Language")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS snomed_ct.Language ("
        "  id uuid NOT NULL,"
        "  effectiveTime integer NOT NULL,"
        "  active boolean NOT NULL,"
        "  moduleId bigint NOT NULL,"
        "  refsetId bigint NOT NULL,"
        "  referencedComponentId bigint NOT NULL,"
        "  acceptabilityId bigint NOT NULL"
        ")"
    )
    cur.execute(
        "CREATE INDEX idx_lang_ref "
        "ON snomed_ct.Language (referencedComponentId) "
    )


def create_refset_extended_map(cur):
    logger.info("Creating table snomed_ct.ExtendedMap")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS snomed_ct.ExtendedMap ("
        "  id uuid NOT NULL, "
        "  effectiveTime integer NOT NULL, "
        "  active boolean NOT NULL, "
        "  moduleId bigint NOT NULL, "
        "  refsetId bigint NOT NULL, "
        "  referencedComponentId bigint NOT NULL, "
        "  mapGroup integer NOT NULL, "
        "  mapPriority integer NOT NULL, "
        "  mapRule text NOT NULL, "  # This is a boolean statement.
        "  mapAdvice text NOT NULL, "
        "  mapTarget text , "
        "  correlationId bigint NOT NULL, "
        "  mapCategoryId bigint NOT NULL"
        ")"
    )
    cur.execute(
        "CREATE INDEX idx_map_cmp "
        "ON snomed_ct.ExtendedMap (referencedComponentId)"
    )


def parse_args():
    description = (
        "Helper script to help create the SNOMED-CT PostgreSQL database from "
        "the RF2 files."
    )

    parser = argparse.ArgumentParser(description=description)

    parent_parser = argparse.ArgumentParser(add_help=False)

    subparser = parser.add_subparsers(
        dest="command",
        help="Either create the configuration file or build the database."
    )
    subparser.required = True

    # Assign this to a variable to add arguments.
    subparser.add_parser(
        "config",
        help="Generate the configuration file need to build the database.",
        parents=[parent_parser]
    )

    build_parser = subparser.add_parser(
        "build",
        help="Build the database from the configuration file.",
        parents=[parent_parser],
    )

    build_parser.add_argument(
        "configuration_file",
        help=("Path to the configuration file (generated using the 'config' "
              "command)."),
        type=str
    )

    args = parser.parse_args()

    if args.command == "config":
        generate_configuration_file()
    elif args.command == "build":
        build_database_from_config(args.configuration_file)


if __name__ == "__main__":
    parse_args()
