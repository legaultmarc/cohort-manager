import unittest
import shutil

import numpy as np

from ..core import CohortManager


def _build_manager():
    return CohortManager("_TestManager")


class TestManager(unittest.TestCase):
    def setUp(self):
        self.tearDown()
        self.manager = _build_manager()

    def tearDown(self):
        try:
            shutil.rmtree(self.manager.name)
        except Exception:
            pass

    def test_set_samples_str(self):
        self.manager.set_samples(["a", "b", "c"])
        self.assertTrue(np.all(
                self.manager.get_samples() == np.array(["a", "b", "c"])
        ))

    def test_set_samples_np(self):
        self.manager.set_samples(np.array(["a", "b", "c"], dtype=np.string_))
        self.assertTrue(np.all(
                self.manager.get_samples() == np.array(["a", "b", "c"])
        ))
