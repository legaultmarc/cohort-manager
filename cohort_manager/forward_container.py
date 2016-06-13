"""
Implementation of the Forward (http://legaultmarc.github.io/forward) phenotype
container on top of the cohort manager.
"""

import shutil
import unittest
import logging

from forward.phenotype.db import PandasPhenotypeDatabase, apply_transformation
from forward.utils import dispatch_methods
from forward.tests.abstract_tests import TestAbstractPhenDB
from six.moves import range
import numpy as np
import scipy.stats

from . import core


logger = logging.getLogger(__name__)


class PhenotypeManagerForwardContainer(PandasPhenotypeDatabase):
    def __init__(self, name, path=None, **kwargs):
        self.manager = core.CohortManager(name, path)
        self.manager_samples = np.array(self.manager.get_samples())

        self._order_is_set = False
        self.variables = None

        self.permutation = None
        self.analyzed_outcomes = []

        dispatch_methods(self, kwargs)

    def get_correlation_matrix(self, names):
        n = len(names)
        mat = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mat[i, j] = 1
                else:
                    mat[i, j] = mat[j, i] = self._get_correlation(names[i],
                                                                  names[j])
        return mat

    def _get_correlation(self, phen1, phen2):
        return scipy.stats.pearsonr(
            self.manager.get_data(phen1),
            self.manager.get_data(phen2)
        )[0]

    def get_phenotype_vector(self, variable, warn=True):
        if not self._order_is_set and warn:
            logger.warning("The order of samples for the database has not "
                           "been set. Make sure that it is consistent with "
                           "the genetic database (consistent order).")

        if not variable.is_variable():
            raise ValueError(
                "'{}' is not a Variable instance (type: {}).".format(
                    variable, type(variable)
                )
            )

        name = variable.name

        try:
            meta = self.manager.get_phenotype(name)
            # Reorder if needed.
            if self.permutation is not None:
                data = self.permutation.get_data(name)
            else:
                data = self.manager.get_data(name)
        except KeyError:
            raise ValueError("'{}' is not in the database.".format(name))

        if meta["variable_type"] == "factor":
            raise ValueError("Forward does not support the analysis of "
                             "factors (it can't represent them using it's "
                             "variable system).")

        if variable.variable_type == "continuous" and variable.transformation:
            data = apply_transformation(variable.transformation, data)

        return data

    def get_phenotypes(self):
        return self.manager.get_phenotypes_list()

    def get_sample_order(self):
        if self.permutation is None:
            return list(self.manager_samples)
        else:
            return list(self.permutation.samples)

    def set_sample_order(self, sequence, allow_subset=False):
        self.permutation = core.Permutation(
            self.manager, sequence, allow_subset
        )


# TESTS
class TestCohortManagerforwardContainer(TestAbstractPhenDB, unittest.TestCase):
    def setUp(self):
        super(TestCohortManagerforwardContainer, self).setUp()

        # Create a dummy phenotype manager.
        manager = core.CohortManager("_TestManager")
        manager.set_samples(["sample_{}".format(i) for i in range(1, 11)])
        manager.add_phenotype(
            name="phenotype1", variable_type="continuous"
        )
        manager.add_phenotype(
            name="phenotype2", variable_type="continuous", parent="phenotype1"
        )
        manager.add_phenotype(
            name="phenotype3", variable_type="discrete",
        )
        manager.add_phenotype(
            name="phenotype4", variable_type="discrete",
        )
        manager.add_phenotype(
            name="phenotype5", variable_type="discrete", parent="phenotype4"
        )

        manager.add_data("phenotype1", np.random.random(10))
        manager.add_data("phenotype2", scipy.stats.norm.rvs(4, 2, size=10))
        manager.add_data("phenotype3", scipy.stats.binom.rvs(1, 0.2, size=10))
        phen4 = scipy.stats.binom.rvs(1, 0.1, size=10)
        manager.add_data("phenotype4", phen4)
        phen5 = phen4.copy()
        for i, value in enumerate(phen4):
            if value == 1:
                phen5[i] = int(np.random.random() < 0.5)
        manager.add_data("phenotype5", phen5)

        manager.commit()
        manager.rebuild_tree()
        manager.validate()

        self.db = PhenotypeManagerForwardContainer("_TestManager")
        self._variables = ["phenotype{}".format(i) for i in range(1, 6)]

    def tearDown(self):
        shutil.rmtree("_TestManager")
