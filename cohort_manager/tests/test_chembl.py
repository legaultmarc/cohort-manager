import unittest
import psycopg2

from ..drugs.chembl import ChEMBL


NO_CHEMBL_MESSAGE = "Local ChEMBL database not installed."
try:
    chembl = ChEMBL()
    CHEMBL_INSTALLED = True
except psycopg2.OperationalError:
    CHEMBL_INSTALLED = False


@unittest.skipIf(not CHEMBL_INSTALLED, NO_CHEMBL_MESSAGE)
class TestChEMBL(unittest.TestCase):
    def test_get_related_drugs(self):
        with ChEMBL() as db:
            # Query on warfarin (parent) returns the two children + parent.
            self.assertEqual(
                db.get_related_drugs(394165),
                {394165, 674723, 674830}
            )

            # Query on a salt return parent.
            self.assertEqual(db.get_related_drugs(674723), {674723, 394165})

    def test_get_parent(self):
        with ChEMBL() as db:
            # 394165 is the parent of 674723.
            self.assertEqual(db.get_parent(674723), 394165)

            # No parent.
            self.assertEqual(db.get_parent(394165), 394165)

    def test_get_children(self):
        with ChEMBL() as db:
            # 674723 has no children (this might fail in future versions).
            self.assertEqual(db.get_children(674723), set())

            # 394165 has 2 children.
            self.assertTrue(674723 in db.get_children(394165))
            self.assertTrue(674830 in db.get_children(394165))

            # Another case (atorvastatin -> atorvastatin calcium).
            self.assertTrue(407354 in db.get_children(417180))

    def test_get_drugs_modulating_protein(self):
        # Take statins.
        protein = "P04035"
        with ChEMBL() as db:
            drugs = db.get_drugs_modulating_protein(protein)

        # Got the results from directly querying chembl_20.
        statins = {9495, 674514, 699412, 1449167, 1123018, 30602, 407354,
                   138562}
        drugs = set(drugs)
        for statin_id in statins:
            self.assertTrue(statin_id in drugs)

    def test_get_drugs_modulating_missing(self):
        with self.assertRaises(ValueError):
            with ChEMBL() as db:
                db.get_drugs_modulating_protein("Q91QZ3")

    def test_get_drugs_modulating_action(self):
        with ChEMBL() as db:
            # Drugs that are agonists for http://www.uniprot.org/uniprot/P14416
            # The D(2) dopamine receptor (multiple actions exist).
            drugs = db.get_drugs_modulating_protein("P14416", action="AGONIST")
        drugs = set(drugs)

        bad_examples = {
            675025, 674658, 674991, 80209, 468656, 364141, 675692, 674937
        }
        for bad in bad_examples:
            self.assertTrue(bad not in drugs)

        good_examples = {
            675038, 1510461, 249814, 547836, 674362, 466038, 258371
        }
        for good in good_examples:
            self.assertTrue(good in drugs)
