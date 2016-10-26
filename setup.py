#!/usr/bin/env python
# -*- coding: utf-8 -*-

# How to build source distribution
# python setup.py sdist --format bztar
# python setup.py sdist --format gztar
# python setup.py sdist --format zip

import os

from setuptools import setup, find_packages
from distutils.core import Extension


MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MICRO)


def write_version_file(fn=None):
    if fn is None:
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join("cohort_manager", "version.py")
        )
    content = ("# THIS FILE WAS GENERATED FROM SETUP.PY\n"
               "cohort_manager_version = \"{version}\"\n")

    a = open(fn, "w")
    try:
        a.write(content.format(version=VERSION))
    finally:
        a.close()


def string_search_extension():
    extension_info = {
        "sources": ["cohort_manager/src/query.cpp",
                    "cohort_manager/src/py_stringsearch.cpp"],
    }
    return Extension("cohort_manager.c_string_search", **extension_info)


def setup_package():
    # Saving the version into a file
    write_version_file()

    setup(
        name="cohort_manager",
        version=VERSION,
        description="Tool to manage collections of phenotype data.",
        long_description=("This package provides an API and a REPL interface "
                          "to access standardized variables."),
        author=u"Marc-André Legault",
        author_email="legaultmarc@gmail.com",
        url="https://github.com/legaultmarc/cohort_manager",
        license="MIT",
        packages=find_packages(exclude=["tests", ]),
        entry_points={
            "console_scripts": [
                "cohort-repl=cohort_manager.scripts.cohort_repl:entry_point",
                "drug-db-builder=cohort_manager.scripts.drug_db_builder:parse_args",
                "cohort-import=cohort_manager.scripts.cohort_import:parse_args",
                "cohort-sim=cohort_manager.scripts.cohort_sim:parse_args",
                "cohort-snomed-ct=cohort_manager.scripts.cohort_snomed_ct:parse_args",
            ],
        },
        classifiers=["Development Status :: 4 - Beta",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "Operating System :: Unix",
                     "Operating System :: MacOS :: MacOS X",
                     "Operating System :: POSIX :: Linux",
                     "Programming Language :: Python",
                     "Programming Language :: Python :: 3",
                     "Topic :: Scientific/Engineering :: Bio-Informatics"],
        test_suite="cohort_manager.tests.test_suite",
        keywords="bioinformatics genomics phewas epidemiology cohort",
        install_requires=["numpy >= 1.8.1", "pandas >= 0.15", "six >= 1.10",
                          "psycopg2 >= 2.6", "scipy >= 0.18",
                          "Unidecode >= 0.4.17"],
        zip_safe=False,
        ext_modules=[
            string_search_extension(),
        ]
    )


if __name__ == "__main__":
    setup_package()
