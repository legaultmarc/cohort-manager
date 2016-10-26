"""
Interface for the ChEMBL database.

This requires a local copy of the (publicly) available PostgreSQL database.
It is based on ChEMBL 20:

ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_20/

"""

from __future__ import division, print_function

import collections
import base64
import logging

import psycopg2


from . import atc
from ..config import configuration


logger = logging.getLogger(__name__)


# Data sources used for synonym search.
_SYNONYM_SOURCES = ("USAN", "USP", "FDA", "TRADE_NAME")


class ChEMBLNotInstalled(Exception):
    def __init__(self):
        self.value = ("The local ChEMBL database is either not installed or "
                      "not properly configured.")

    def __str__(self):
        return self.value


class ChEMBL(object):
    """Class to facilitate ChEMBL queries.

    This class is configured using the configuration file (see the
    cohort_manager.config module for more information).

    host: The hostname (default 'localhost').
    port: The port (default '5432').
    name: The database name.
    username: The PostgreSQL database username.
    password: The PostgreSQL database password.
    b64_password: A base64 encoded PostgreSQL database password (for
                  people who don't want to set passwords in configuration
                  files as plain text).

    The PostgreSQL database needed for this class to work can be downloaded
    from the ChEMBL FTP server:

    ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/

    Use the PostgreSQL ChEMBL 21 release file (chembl_21_postgresql.tar.gz)
    file and follow the instructions from the downloaded file.

    """
    def __init__(self):
        try:
            conf = configuration.chembl

            password = conf["password"]
            if (not password) and conf["b64_password"]:
                password = base64.b64decode(
                    conf["b64_password"]
                ).decode("utf-8")

            self.con = psycopg2.connect(
                database=conf["name"], user=conf["username"],
                password=password, host=conf["host"], port=conf["port"]
            )
        except psycopg2.OperationalError:
            raise ChEMBLNotInstalled()

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

    def get_active_compound(self, molregno):
        """Get the molregno of the active compound for a prodrug."""
        res = self.execute(
            "SELECT MOLREGNO, ACTIVE_MOLREGNO "
            "FROM MOLECULE_HIERARCHY WHERE MOLREGNO=%s", (molregno, )
        )
        if not res:
            return molregno
        assert len(res) == 1
        return res[0][1]

    def get_parent(self, molregno):
        """Get the molregno of the parent drug.

        If the drug has no distinct parent, simply returns the provided
        molregno.
        """
        molecule_hierarchy = self.execute(
            "SELECT MOLREGNO, PARENT_MOLREGNO "
            "FROM MOLECULE_HIERARCHY WHERE MOLREGNO=%s", (molregno, )
        )
        if not molecule_hierarchy:
            return molregno
        assert len(molecule_hierarchy) == 1
        return molecule_hierarchy[0][1]

    def get_children(self, molregno):
        """Get the molregno(s) of the children drug(s).

        If the drug has no children, simply returns an empty set.

        """
        res = self.execute(
            "SELECT MOLREGNO, PARENT_MOLREGNO "
            "FROM MOLECULE_HIERARCHY WHERE PARENT_MOLREGNO=%s", (molregno, )
        )
        if not res:
            return set()

        return {i[0] for i in res if i[0] != molregno}

    def get_related_drugs(self, molregno):
        """Get drugs that are related according to the molecule hierarchy.

        :param molregno: The molregno.
        :type molregno: int

        :returns: A set of related molregnos.
        :rtype: set

        """
        res = self.execute(
            "SELECT MOLREGNO, PARENT_MOLREGNO FROM MOLECULE_HIERARCHY "
            "WHERE MOLREGNO=%s OR PARENT_MOLREGNO=%s", (molregno, molregno)
        )
        out = set()
        for molregno, parent in res:
            out.add(molregno)
            out.add(parent)
        return out

    def get_drugs_with_atc(self, atc_code):
        """Returns a list of drugs with the provided ATC code.

        This function is not aware of the hiarachical relationship.

        A ValueError will be raised if there are no drugs with the provided
        code.

        """
        # Use ChEMBL to get a list of drugs with a matching ATC code.
        atc_level = atc.get_atc_code_level(atc_code)
        sql = ("SELECT mac.MOLREGNO FROM MOLECULE_ATC_CLASSIFICATION mac, "
               "ATC_CLASSIFICATION ac WHERE ac.LEVEL{level}=%s AND "
               "mac.LEVEL5=ac.LEVEL5".format(level=atc_level))

        results = self.execute(sql, (atc_code, ))
        if results:
            return [i[0] for i in results]

        raise ValueError("Could not find drugs with ATC code '{}' (level {})"
                         "".format(atc_code, atc_level))

    def get_drugs_modulating_protein(self, uniprot_id, action=None):
        """Returns a list of drugs modulating the provided protein.

        This function is not aware of the hierarchical relationship.

        A ValueError will be raised if there are no drugs modulating the
        provided target.

        """
        sql = (
            "SELECT DISTINCT DM.MOLREGNO "
            "FROM COMPONENT_SEQUENCES as CS "
            " JOIN TARGET_COMPONENTS as TC "
            "   ON CS.COMPONENT_ID=TC.COMPONENT_ID "
            " JOIN TARGET_DICTIONARY as TD "
            "   ON TC.TID=TD.TID "
            " JOIN DRUG_MECHANISM as DM "
            "   ON TD.TID=DM.TID "
            "WHERE CS.ACCESSION=%s"
        )
        if action is not None:
            sql += " AND DM.ACTION_TYPE=%s"
            params = (uniprot_id, action)
        else:
            params = (uniprot_id, )

        ids = self.execute(sql, params)
        if not ids:
            raise ValueError("Could not find any drugs modulating '{}'."
                             "".format(uniprot_id))

        return [i[0] for i in ids]

    def get_drug_info(self, molregno):
        """Returns information on a drug.

        :param molregno: The molregno.
        :type molregno: int

        :returns: A dict of aggregated drug information.
        :rtype: dict

        The function is used as-is for the REPL's 'drug_info' function.
        Most fields are directly populated from the database, but others
        are generated by querying related tables.

        """
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

        # ATC information.
        out["ATC"] = self._get_atc(molregno)

        # Mechanism information.
        out["Mechanism"] = self._get_mechanism(molregno)

        # Parent information.
        parent = self.get_parent(molregno)
        out["parent"] = None if molregno == parent else parent
        if out["parent"]:
            # Inherit the ATC information.
            out["ATC_inherited"] = self._get_atc(out["parent"])
            del out["ATC"]

        # Children information.
        out["children"] = tuple(self.get_children(molregno))

        return out

    def _get_mechanism(self, molregno):
        # Get mechanism of action information.
        mechanism_fields = ("mechanism_of_action", "action_type",
                            "direct_interaction", "mechanism_comment",
                            "selectivity_comment", "binding_site_comment")
        mechanism = self.execute(
            "SELECT tid, {} FROM drug_mechanism WHERE molregno=%s".format(
                ", ".join(mechanism_fields)
            ),
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
        atc_fields = ("level5", "who_id", "who_name", "level1_description",
                      "level2_description", "level3_description",
                      "level4_description")

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
        fields = ("tid", "pref_name", "organism")
        targets = self.execute(
            "SELECT {} FROM target_dictionary "
            "WHERE TID=%s".format(", ".join(fields)),
            (tid, )
        )
        targets = [dict(zip(fields, t)) for t in targets]

        for target_info in targets:
            target_info["components"] = self._get_target_components(tid)

        return targets

    def _get_target_components(self, tid):
        fields = ("component_type", "accession", "description")
        components = self.execute(
            "SELECT cs.{} "
            "FROM component_sequences cs, target_components tc "
            "WHERE tc.component_id=cs.component_id AND "
            "tc.tid=%s".format(", cs.".join(fields)),
            (tid, )
        )
        components = [dict(zip(fields, component)) for component in components]
        return components
