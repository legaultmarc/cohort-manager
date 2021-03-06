"""
Bindings to the genetest package used for statistical analyses.
"""

from ..core import CohortManager

from genetest.phenotypes.core import PhenotypesContainer


class CohortManagerContainer(PhenotypesContainer):
    def __init__(self, name, path=None):
        if path is None:
            path = ""
        self.manager = CohortManager(name, path)
        self.samples = None

    def close(self):
        self.manager.close()

    def keep_samples(self, keep):
        self.samples = keep

    def is_repeated(self):
        """Check if phenotype contains repeated measurements."""
        raise NotImplementedError()

    def get_nb_variables(self):
        """Returns the number of variables."""
        return self.manager.get_number_phenotypes()

    def get_nb_samples(self):
        """Returns the number of samples."""
        return self.manager.n

    def get_phenotypes(self, li=None):
        """Returns a dataframe of phenotypes.
        Returns:
            pandas.DataFrame: A dataframe containing the phenotypes (with the
            sample IDs as index).
        """
        if li is None:
            li = self.manager.get_phenotypes_list()

        df = self.manager.get_dataframe(li)

        # Subset if needed.
        if self.samples is not None:
            return df.loc[self.samples, :]
        else:
            return df
