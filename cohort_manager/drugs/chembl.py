"""
Interface for the ChEMBL database.

This requires a local copy of the (publicly) available PostgreSQL database.
It is based on ChEMBL 20:

ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_20/

"""

from __future__ import division, print_function

import functools
import base64
import os
import logging
import functools
import multiprocessing

import psycopg2
import numpy as np


logger = logging.getLogger(__name__)


DB_HOST = os.environ.get("DB_CHEMBL_HOST", "localhost")
DB_PORT = os.environ.get("DB_CHEMBL_PORT", 5432)
DB_NAME = os.environ.get("DB_CHEMBL_NAME")
DB_USERNAME = os.environ.get("DB_CHEMBL_USERNAME")
DB_PASSWORD = os.environ.get("DB_CHEMBL_PASSWORD")
if not DB_PASSWORD:
    try:
        DB_PASSWORD = base64.b64decode(
            os.environ.get("DB_CHEMBL_B64_PASSWORD")
        ).decode("utf-8")
    except TypeError:
        pass

# Data sources used for synonym search.
_SYNONYM_SOURCES = ("USAN", "USP", "FDA", "TRADE_NAME")


class Drug(object):
    def __init__(self, molregno):
        self.molregno = molregno
        with ChEMBL() as db:
            self.data = db.execute("SELECT * FROM molecule_dictionary WHERE "
                                   "molregno=%s", (molregno, ))
        print(self.data)


class ChEMBL(object):
    def __init__(self):
        self._get_con = functools.partial(
            psycopg2.connect, database=DB_NAME, user=DB_USERNAME,
            password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )

        self.con = self._get_con()
        self.cur = self.con.cursor()
        self._cache = {}


    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def execute(self, sql, params=None):
        if params is None:
            params = tuple()

        self.cur.execute(sql, params)
        return self.cur.fetchall()

    def get_all_drugs(self):
        """Get a list of relevant drug names from ChEMBL.

        :returns: A list of tuple (molregno, drug_name)
        :rtype: list

        This uses both the preferred names and the synonyms.

        """
        drugs = self.execute("SELECT molregno, UPPER(pref_name) FROM "
                             "molecule_dictionary WHERE pref_name IS NOT NULL")

        sql = ("SELECT molregno, UPPER(synonyms), syn_type "
               "FROM molecule_synonyms "
               "WHERE syn_type IN {}").format(_SYNONYM_SOURCES)

        return list(set(drugs) | {(i[0], i[1]) for i in self.execute(sql)})
