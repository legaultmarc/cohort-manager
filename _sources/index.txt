.. CohortManager documentation master file, created by
   sphinx-quickstart on Fri Jun 10 15:42:56 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CohortManager Documentation
=============================

Introduction
-------------

CohortManager is a tool used to manage large collections of phenotype and
pharmacotherapy data. It was designed to facilitate high throughput phenomic
studies and it integrates with `forward <github.com/legaultmarc/forward>`_
which can be used for statistical testing.

The following examples shows some of the interesting features provided with
the manager:

- A REPL (read-eval-print-loop): A conventient and powerful interface that
  makes it easy to interact with the manager without knowing about the Python
  API.
- Possibility to generate summary statistics and update metadata.
- A "virtual variable" functionality which makes it easy to define new
  variables that can be somewhat complex combinations of existing phenotypes.

.. raw:: html

    <script type="text/javascript" src="https://asciinema.org/a/7ztnw5m1zqfrw4qqk51fcrmoh.js" id="asciicast-7ztnw5m1zqfrw4qqk51fcrmoh" async></script>

The CohortManager also comes with a pharmacotherapy module that is integrated
with ChEMBL.


Contents:

.. toctree::
    :maxdepth: 2

    api.rst

    type_system.rst

    importing.rst

    hierarchy.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

