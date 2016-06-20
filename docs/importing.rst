Importing data
===============

Importing data is done in two steps. First, a parsing step where freetext files
(`e.g.` CSV files) are read and the data type of columns is inferred. This
step will generate an Excel file containing the following columns:

group
    Group number for the variable. This group number is not stored and it
    is only used to display similar variables together to facilitate curation.
name
    The column name in the data file.
parent
    No inference is done on this field, but it can be filled manually to
    represent hierarchy between variables. For more information on how
    hierarchy is used throughout CohortManager, see :doc:`./hierarchy`.

.. code::

    usage: cohort-import [-h] {parse,build} ...

    Helper script to help import phenotype data into a cohort.

    positional arguments:
      {parse,build}  Either parse a phenotype input file or build the cohort
                     managerdatabase.
        parse        Parse phenotype input file.
        build        Fill the cohort manager database given the import description
                     file.

    optional arguments:
      -h, --help     show this help message and exit
