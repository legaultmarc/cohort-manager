"""
Interface to SNOMED-CT.
"""

import logging
import base64

import psycopg2

from .config import configuration


logger = logging.getLogger(__name__)


SNOMED_CT_CORE = 900000000000207008
SYNONYM = 900000000000013009
FULLY_SPECIFIED_NAME = 900000000000003001
PREFERRED = 900000000000548007

# Relationship concepts
IS_A = 116680003

# Mapping concepts
ICD_10_MAP = 447562003
SOURCE_PROPERLY_MAPPED = 447637006


class SNOMEDCTNotInstalled(Exception):
    def __init__(self):
        self.value = ("The local SNOMED-CT database is either not installed "
                      "or not properly configured.")

    def __str__(self):
        return self.value


class SnomedCT(object):
    """Minimalist API over the SNOMED-CT database.

    First, it is necessary to build the PostgreSQL database using the
    `cohort-snomed-ct` script.

    """
    def __init__(self):
        cfg = configuration.snomed_ct

        password = cfg["password"]
        if (not password) and cfg["b64_password"]:
            password = base64.b64decode(
                cfg["b64_password"]
            ).decode("utf-8")

        try:
            self.con = psycopg2.connect(
                database=cfg["name"], user=cfg["username"], password=password,
                host=cfg["host"], port=cfg["port"]
            )
            logger.debug(
                "Connected to database {} (on host {}:{}) with user {}"
                "".format(cfg["name"], cfg["host"], cfg["port"],
                          cfg["username"])
            )
        except psycopg2.OperationalError:
            raise SNOMEDCTNotInstalled()

        self.cur = self.con.cursor()

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_term_from_conceptid(self, concept_id):
        """Get the term (text) associated with a concept id."""
        sql = (
            "SELECT DISTINCT(term) FROM snomed_ct.Description "
            "WHERE conceptId=%s AND active"
        )
        self.cur.execute(sql, (concept_id, ))
        return [i[0] for i in self.cur]

    def search_concept(self, s, fully_specified_name=False):
        """Search concepts using a string.

        Use '%' as a wildcard if needed.

        The concept_id, term, type (description) and effectiveTime are
        returned. Searches are case-insensitive.

        This search is done in the SNOMED-CT Core module.
        If fully_specified_name is set to True, only such concepts will
        be queried.

        """
        sql = (
            "SELECT d1.conceptId, d1.term, d1.typeId, d2.term "
            "FROM "
            "  snomed_ct.description d1 "
            "  INNER JOIN snomed_ct.description d2 ON d1.typeId=d2.conceptId "
            "WHERE "
            "  (SELECT active FROM snomed_ct.concept "
            "   WHERE id=d1.conceptId "
            "   ORDER BY effectivetime DESC LIMIT 1) AND "
            "  d1.active AND "
            "  LOWER(d1.term) LIKE %s AND "
            "  d1.moduleId=%s"
        )
        args = [s.lower(), SNOMED_CT_CORE]

        if fully_specified_name:
            sql += " AND d1.typeId=%s"
            args.append(FULLY_SPECIFIED_NAME)

        self.cur.execute(sql, tuple(args))
        return list(self.cur)

    def get_relationship(self, concept_id):
        """Get all the relationships targeting the given concept.

        Returns tuples of (sourceId, typeId, destinationId)

        """
        sql = (
            "SELECT "
            "  sourceId, "
            "  typeId, "
            "  destinationId "
            "FROM "
            "  snomed_ct.Relationship r, "
            "  snomed_ct.Concept cct_t, "
            "  snomed_ct.Concept cct_d "
            "WHERE "
            "  r.typeId=cct_t.id AND "
            "  r.destinationId=cct_d.id AND "
            "  r.active AND cct_t.active AND cct_d.active AND "
            "  sourceId=%s"
        )
        self.cur.execute(sql, (concept_id, ))
        return list(self.cur)

    def plot_relationship(self, root):
        """Non-recursively print the DOT language description of the
        relationships originating from a given root concept.

        """
        edges = set(self.get_relationship(root))

        print("digraph g {")
        for start, t, end in edges:
            desc = {
                "start": self.get_term_from_conceptid(start)[0],
                "type": self.get_term_from_conceptid(t)[0],
                "end": self.get_term_from_conceptid(end)[0],
            }

            print('    "{start}" -> "{end}"[label="{type}"];'.format(**desc))
        print("}")

    def get_preferred(self, concept_id):
        """Get the preferred term for a given concept id."""
        sql = (
            "SELECT DISTINCT d.conceptid, d.term "
            "FROM "
            "  snomed_ct.Description d, "
            "  snomed_ct.Language l, "
            "  snomed_ct.Concept c "
            "WHERE "
            "  d.conceptId=%s AND "
            "  c.id=d.conceptId AND "
            "  d.id=l.referencedComponentId AND "
            "  c.active AND "
            "  d.active AND "
            "  l.active AND "
            "  d.typeId=%s AND "
            "  l.acceptabilityId=%s"
        )
        self.cur.execute(
            sql, (concept_id, SYNONYM, PREFERRED)
        )
        return list(self.cur)

    def get_all_disorder_synonyms(self):
        """Get the synonyms for all concepts with the 'disorder' semantic tag.

        """
        # Join on d2 is used to restrict to entries with the 'disorder'
        # semantic tag.
        sql = (
            "SELECT DISTINCT d1.conceptId, d1.term "
            "FROM "
            "  snomed_ct.Description d1, "
            "  snomed_ct.Description d2, "
            "  snomed_ct.Concept c "
            "WHERE "
            "  c.id = d1.conceptId AND "
            "  d1.typeId=%s AND "
            "  d1.conceptId=d2.conceptId AND "
            "  d2.term LIKE '%%(disorder)' AND "
            "  c.active AND "
            "  d1.active AND "
            "  d2.active"
        )
        self.cur.execute(sql, (SYNONYM, ))
        return self.cur.fetchall()

    def get_synonyms(self, concept_id):
        """Get all synonyms for a concept_id."""
        sql = (
            "SELECT DISTINCT term "
            "FROM snomed_ct.Description "
            "WHERE "
            "  conceptId=%s AND "
            "  typeId=%s AND "
            "  active"
        )
        self.cur.execute(sql, (concept_id, SYNONYM))
        return [i[0] for i in self.cur]

    def get_description(self, concept_id):
        """Get information on a concept id."""
        cols = (
            "id", "effectivetime", "active", "moduleid", "conceptid",
            "languagecode", "typeid", "term", "casesignificanceid"
        )

        sql = (
            "SELECT * "
            "FROM snomed_Ct.Description "
            "WHERE "
            "  conceptId=%s"
        )
        self.cur.execute(sql, (concept_id, ))
        out = []
        for tu in self.cur.fetchall():
            out.append({k: v for k, v in zip(cols, tu)})
        return out

    def get_fully_specified_name(self, concept_id):
        """Get the fully specified name for a concept_id."""
        sql = (
            "SELECT DISTINCT term FROM snomed_ct.Description "
            "WHERE "
            "  conceptId=%s AND "
            "  active AND "
            "  typeId=%s"
        )
        self.cur.execute(sql, (concept_id, FULLY_SPECIFIED_NAME))
        return [i[0] for i in self.cur.fetchall()]

    def get_icd10_code(self, concept_id):
        """Get the ICD10 code from a disorder concept id."""
        sql = (
            "SELECT DISTINCT mapTarget, mapAdvice "
            "FROM snomed_ct.extendedmap "
            "WHERE "
            "  mapCategoryId=%s AND "
            "  refsetid=%s AND "
            "  active AND "
            "  mapTarget IS NOT NULL AND "
            "  referencedComponentId=%s"
        )
        self.cur.execute(sql, (SOURCE_PROPERLY_MAPPED, ICD_10_MAP, concept_id))
        return list(self.cur)
