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
import Levenshtein  # pip install python-Levenshtein
import numpy as np


from .tokens import tokenize, Token, longest_name


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
_SIMILARITY_THRESHOLD = 0.7


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

    def search_compounds(self, li):
        """Query ChEMBL for a list of compounds."""

        n_cpus = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(n_cpus)

        tokens = [tokenize(s) for s in li]
        longest_names = [longest_name(i) for i in tokens if i is not None]
        longest_names = list({i.value for i in longest_names if i is not None})

        # Batch query ChEMBL.
        matchers = [
            self._compound_match_preferred_name,
            functools.partial(self._compound_match_preferred_name, fuzzy=True),
            self._compound_match_synonym,
            functools.partial(self._compound_match_synonym, fuzzy=True),
        ]

        matches = []
        for f in matchers:
            matches.extend(zip(longest_names, pool.map(f, longest_names)))

        return matches

    @staticmethod
    def _jaro(s, li):
        """Return the index in li of the word with the largest Jaro similarity
        with s.

        """
        scores = list(map(functools.partial(Levenshtein.jaro, s), li))
        max_idx = np.argmax(scores)
        return max_idx, scores[max_idx]

    def _compound_match_synonym(self, s, fuzzy=False):
        """Match a compound by synonym name.

        Returns a tuple of molregno, synonyms, syn_type.

        """
        sql = ("SELECT molregno, synonyms, syn_type FROM MOLECULE_SYNONYMS "
               "WHERE syn_type IN {} "
               "AND UPPER(SYNONYMS)".format(_SYNONYM_SOURCES))

        s = s.upper()
        if fuzzy:
            sql += " LIKE %s"
            args = ("%{}%".format(s), )
        else:
            sql += "=%s"
            args = (s, )

        with self._get_con() as con:
            with con.cursor() as cur:
                cur.execute(sql, args)
                return cur.fetchall()

    def _compound_match_preferred_name(self, s, fuzzy=False):
        """Match a compound by preferred name.

        Returns a tuple of molregno, pref_name

        """
        sql = ("SELECT molregno, pref_name FROM molecule_dictionary WHERE "
               "UPPER(pref_name)")

        s = s.upper()
        if fuzzy:
            sql += " LIKE %s"
            args = ("%{}%".format(s), )
        else:
            sql += "=%s"
            args = (s, )

        with self._get_con() as con:
            with con.cursor() as cur:
                cur.execute(sql, args)
                return cur.fetchall()
