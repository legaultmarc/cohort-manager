"""
Interface for the ChEMBL database.

This requires a local copy of the (publicly) available PostgreSQL database.
It is based on ChEMBL 20:

ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_20/

"""

from __future__ import division, print_function

import collections
import functools
import base64
import os
import re
import logging

import psycopg2


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

    def get_preferred_drugs(self):
        """Get a list of preferred drug names from ChEMBL.

        :returns: A list of tuple (molregno, drug_name)
        :rtype: list

        """
        return self.execute("SELECT molregno, UPPER(pref_name) FROM "
                            "molecule_dictionary WHERE pref_name IS NOT NULL")

    def get_synonyms(self):
        """Get a list of synonym drug names from ChEMBL.

        :returns: A list of tuple (molregno, drug_name, synonym_type)
        :rtype: list

        The queried synonym types are defined in `_SYNONYM_SOURCES`.

        """
        return self.execute("SELECT molregno, UPPER(synonyms), syn_type "
                            "FROM molecule_synonyms "
                            "WHERE syn_type IN {}".format(_SYNONYM_SOURCES))

    def get_all_drugs(self):
        """Get a list of relevant drug names from ChEMBL.

        :returns: A list of tuple (molregno, drug_name)
        :rtype: list

        This uses both the preferred names and the synonyms.

        """
        return list(set(self.get_preferred_drugs()) |
                    {(i[0], i[1]) for i in self.get_synonyms()})

    def get_parent(self, molregno):
        """Get the molregno of the parent drug.

        If the drug has no distinct parent, simply returns the provided
        molregno.
        """
        molecule_hierarchy = self.execute(
            "SELECT MOLREGNO, PARENT_MOLREGNO, ACTIVE_MOLREGNO "
            "FROM MOLECULE_HIERARCHY WHERE MOLREGNO=%s", (molregno, )
        )
        if not molecule_hierarchy:
            return molregno
        assert len(molecule_hierarchy) == 1
        return molecule_hierarchy[0][1]

    def get_drugs_with_atc(self, atc_code):
        # Use ChEMBL to get a list of drugs with a matching ATC code.
        levels = {
            "^[ABCDGHJLMNPRSV]$": 1,
            "^[ABCDGHJLMNPRSV][0-9]{2}$": 2,
            "^[ABCDGHJLMNPRSV][0-9]{2}[A-Z]$": 3,
            "^[ABCDGHJLMNPRSV][0-9]{2}[A-Z]{2}$": 4,
            "^[ABCDGHJLMNPRSV][0-9]{2}[A-Z]{2}[0-9]{2}$": 5,
        }

        matched_level = None
        for regex, level in levels.items():
            if re.match(regex, atc_code.upper()):
                matched_level = level

        if not matched_level:
            raise ValueError("Could not parse ATC code '{}'.".format(atc_code))

        sql = ("SELECT mac.MOLREGNO FROM MOLECULE_ATC_CLASSIFICATION mac, "
               "ATC_CLASSIFICATION ac WHERE ac.LEVEL{level}=%s AND "
               "mac.LEVEL5=ac.LEVEL5".format(level=matched_level))

        results = self.execute(sql, (atc_code, ))
        if results:
            return [i[0] for i in results]

        raise ValueError("Could not find drugs with ATC code '{}' (level {})"
                         "".format(atc_code, matched_level))

    def get_drug_info(self, molregno):
        """Returns information on a drug."""
        out = collections.OrderedDict()

        # Get molecule information.
        molecule_dictionary_fields = (
            "PREF_NAME", "ORAL", "TOPICAL", "PRODRUG", "FIRST_IN_CLASS"
        )
        md_info = self.execute(
            "SELECT {} FROM molecule_dictionary WHERE "
            "molregno=%s".format(", ".join(molecule_dictionary_fields)),
            (molregno, )
        )

        assert len(md_info) == 1
        md_info = md_info.pop()
        md_info = dict(zip(molecule_dictionary_fields, md_info))
        out["molecule_information"] = md_info

        # ATC Information.
        out["ATC"] = self._get_atc(molregno)

        # Mechanism information.
        out["Mechanism"] = self._get_mechanism(molregno)

        return out

    def _get_mechanism(self, molregno):
        # Get mechanism of action information.
        mechanism_fields = ("Mechanism of action", "Action type",
                            "Direct interaction", "Mechanism comment",
                            "Selectivity comment", "Binding site comment")
        mechanism = self.execute(
            "SELECT tid, mechanism_of_action, action_type, direct_interaction,"
            "  mechanism_comment, selectivity_comment, binding_site_comment "
            "FROM drug_mechanism WHERE molregno=%s",
            (molregno, )
        )
        results = []
        for result in mechanism:
            element = dict(zip(mechanism_fields, result[1:]))
            element["target_info"] = self._get_target_info(result[0])
            results.append(element)

        return results

    def _get_atc(self, molregno):
        # Get ATC information.
        atc_fields = ("level5", "WHO (ID)", "WHO (name)", "desc_level1",
                      "desc_level2", "desc_level3", "desc_level4")

        atc = self.execute(
            "SELECT MAC.LEVEL5, AC.WHO_ID, AC.WHO_NAME,"
            "  AC.LEVEL1_DESCRIPTION, AC.LEVEL2_DESCRIPTION,"
            "  AC.LEVEL3_DESCRIPTION, AC.LEVEL4_DESCRIPTION "
            "FROM MOLECULE_ATC_CLASSIFICATION MAC, "
            "ATC_CLASSIFICATION AC "
            "WHERE MAC.LEVEL5=AC.LEVEL5 AND MAC.MOLREGNO=%s",
            (molregno, )
        )
        results = []
        for result in atc:
            results.append(dict(zip(atc_fields, result)))
        return results

    def _get_target_info(self, tid):
        targets = self.execute(
            "SELECT tid, pref_name, organism FROM target_dictionary "
            "WHERE TID=%s",
            (tid, )
        )
        targets = [dict(zip(("TID", "Pref. name", "Organism"), t)) for
                   t in targets]

        for target_info in targets:
            target_info["components"] = self._get_target_components(tid)

        return targets

    def _get_target_components(self, tid):
        components = self.execute(
            "SELECT cs.component_type, cs.accession, cs.description "
            "FROM component_sequences cs, target_components tc "
            "WHERE tc.component_id=cs.component_id AND "
            "tc.tid=%s",
            (tid, )
        )
        components = [
            dict(zip(("Type", "Accession", "Description"), component))
            for component in components
        ]
        return components
