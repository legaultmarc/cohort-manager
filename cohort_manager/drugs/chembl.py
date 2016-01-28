"""
Interface for the ChEMBL database.

This requires a local copy of the (publicly) available PostgreSQL database.
It is based on ChEMBL 20:

ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_20/

"""

from __future__ import division

import functools
import base64
import os
import re
import logging

import psycopg2
import Levenshtein  # pip install python-Levenshtein
import numpy as np


logger = logging.getLogger(__name__)


DB_HOST = os.environ.get("DB_CHEMBL_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", 5432)
DB_NAME = os.environ.get("DB_CHEMBL_NAME")
DB_USERNAME = os.environ.get("DB_CHEMBL_USERNAME")
DB_PASSWORD = os.environ.get("DB_CHEMBL_PASSWORD")
if not DB_PASSWORD:
    try:
        DB_PASSWORD = base64.b64decode(
            os.environ.get("DB_CHEMBL_B64_PASSWORD")
        )
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

        print self.data


class ChEMBL(object):
    def __init__(self):
        self.con = psycopg2.connect(database=DB_NAME, user=DB_USERNAME,
                                    password=DB_PASSWORD, host=DB_HOST,
                                    port=DB_PORT)
        self.cur = self.con.cursor()

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

    def compound_search(self, s):
        """Search for a compound using the ChEMBL known synonyms."""

        # Remove non alphanumeric characters from search string.
        s = re.sub(r"([^\s\w])+", "", s.upper())

        # Try for an exact match in preferred names.
        preferred_match = self._compound_match_preferred_name(s, fuzzy=False)
        unique_ids = {i[0] for i in preferred_match}
        if len(unique_ids) == 1:
            return Drug(unique_ids.pop())
        elif len(unique_ids) > 1:
            logger.warning(
                "{} possible matches (using preferred names).".format(
                    len(unique_ids)
                )
            )
            max_idx, max_score = ChEMBL._jaro(
                s, [i[1] for i in preferred_match]
            )
            if max_score > _SIMILARITY_THRESHOLD:
                return Drug(preferred_match[max_idx][0])

        # Try for an exact match in synonyms or a partial match in synonyms.
        fuzzy = False
        synonyms = self._compound_match_synonym(s, fuzzy)
        unique_ids = {i[0] for i in synonyms}
        if len(synonyms) < 1:
            fuzzy = True
            synonyms = self._compound_match_synonym(s, fuzzy)
            unique_ids = {i[0] for i in synonyms}

        if len(synonyms) < 1:
            logger.warning("Could not find drug: '{}'.".format(s))
            return

        if len(unique_ids) == 1:
            return Drug(unique_ids.pop())
        else:
            logger.warning(
                "{} possible matches for '{}' (fuzzy={}).".format(
                    len(unique_ids), s, fuzzy
                )
            )
            max_idx, max_score = ChEMBL._jaro(
                s, [i[1] for i in synonyms]
            )
            if max_score > _SIMILARITY_THRESHOLD:
                return Drug(synonyms[max_idx][0])

            logger.warning(
                "Could not find drug: '{}' (best match's jaro is under the "
                "minimum threshold: {:.3f}<{:.3f})".format(
                    s, max_score, _SIMILARITY_THRESHOLD
                )
            )
            return

    @staticmethod
    def _jaro(s, li):
        """Return the index in li of the word with the largest Jaro similarity
        with s.

        """
        scores = map(functools.partial(Levenshtein.jaro, s), li)
        max_idx = np.argmax(scores)
        return max_idx, scores[max_idx]

    def _compound_match_synonym(self, s, fuzzy=False):
        """Match a compound by synonym name.

        Returns tuples of molregno, synonyms, syn_type.

        """
        sql = ("SELECT molregno, synonyms, syn_type FROM MOLECULE_SYNONYMS "
               "WHERE syn_type IN {} "
               "AND UPPER(SYNONYMS)".format(_SYNONYM_SOURCES))

        if fuzzy:
            sql += " LIKE '%{}%'".format(s)
            self.cur.execute(sql)
        else:
            sql += "=%s"
            self.cur.execute(sql, (s,))

        return self.cur.fetchall()

    def _compound_match_preferred_name(self, s, fuzzy=False):
        """Match a compound by preferred name.

        Returns tuples of molregno, pref_name

        """
        sql = ("SELECT molregno, pref_name FROM molecule_dictionary WHERE "
               "UPPER(pref_name)")

        if fuzzy:
            sql += " LIKE '%{}%'".format(s)
            params = tuple()
        else:
            sql += "=%s"
            params = (s, )

        self.cur.execute(sql, params)
        return self.cur.fetchall()
