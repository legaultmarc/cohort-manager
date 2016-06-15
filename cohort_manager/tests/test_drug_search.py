import shutil
import unittest

from ..core import CohortManager
from ..drugs import drug_search as ds
from ..drugs.chembl import ChEMBL, ChEMBLNotInstalled


NO_CHEMBL_MESSAGE = "Local ChEMBL database not installed."
try:
    chembl = ChEMBL()
    CHEMBL_INSTALLED = True
except ChEMBLNotInstalled:
    CHEMBL_INSTALLED = False


class TestDrugSearch(unittest.TestCase):
    def setUp(self):
        self.tearDown()
        self.manager = CohortManager("_TestManager")

    def tearDown(self):
        try:
            shutil.rmtree("_TestManager")
        except FileNotFoundError:
            pass

    @unittest.skipIf(not CHEMBL_INSTALLED, NO_CHEMBL_MESSAGE)
    def test_find_drugs_in_query(self):
        #                             012345678901 align from 0 to 11
        #                             MMMMMMMMMMMM score: 12 * 3 = 36
        res = ds.find_drugs_in_query("atorvastatin")
        self.assertEqual(res, [(417180, "ATORVASTATIN", 36, 0, 11)])

    @unittest.skipIf(not CHEMBL_INSTALLED, NO_CHEMBL_MESSAGE)
    def test_find_drugs_spell(self):
        #                             0123456789012 align from 1 to 12
        #                             -MMMMMXMMMMMM score: 11 * 3 - 1 = 32
        res = ds.find_drugs_in_query("satorvzstatin")
        self.assertEqual(res, [(417180, "ATORVASTATIN", 32, 1, 12)])

    @unittest.skipIf(not CHEMBL_INSTALLED, NO_CHEMBL_MESSAGE)
    def test_find_drugs_combination(self):
        res = ds.find_drugs_in_query("tylenol/codein")
        self.assertTrue(16450 in [i[0] for i in res])  # Acetaminophen.
        self.assertTrue(6167 in [i[0] for i in res])  # Codeine.
        self.assertTrue(len(res) == 2)

    @unittest.skipIf(not CHEMBL_INSTALLED, NO_CHEMBL_MESSAGE)
    def test_find_drugs_in_queries(self):
        # TODO
        return
        queries = (
            "LIPITOR", "CRESTOR", "COUMADIN", "SYNTHROID", "PLAVIX", "NORVASC",
            "BISOPROLOL", "ALTACE", "METFORMIN", "METOPROLOL", "NEXIUM",
            "ATIVAN", "ATORVASTATIN", "COVERSYL", "PANTOPRAZOLE", "ATACAND",
            "AMLODIPINE", "FUROSEMIDE", "DIOVAN", "ATENOLOL", "ASPIRINE",
            "ACTONEL", "IMDUR", "AVAPRO", "CELEBREX", "RAMIPRIL", "FLOMAX",
            "GLUCOPHAGE", "CALCIUM", "LANOXIN", "COZAAR", "LOPRESSOR",
            "MICARDIS", "LASIX", "NASONEX", "ALLOPURINOL", "AVODART",
            "NITRO SPRAY", "SANDOZ BISOPROLOL", "LYRICA"
        )

        answers = (
            417180, 1123018, 674830, 1376011, 598402, 418414, 27417, 189203,
            369464, 789, 1418084, 16959, 417180, 250508, 419598, 116349,
            418414, 2046, 139045, 1279, 1280, 494630, 263453, 421066, 18694,
            189203, 61181, 547681, 1418072, 581814, 109797, 705951, 116949,
            2046, 187696, 395607, 674920, 37493, 27417, 136161
        )

        results = ds.find_drugs_in_queries(queries)
        results = ds.fix_hierarchical_matches(queries, results)
        for i, res in enumerate(results):
            print(answers[i], res)
            self.assertTrue(answers[i] in [i[0] for i in res])
