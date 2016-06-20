Hierarchy between variables
============================

Collections of phenotypes often exhibit a hierarchical structure. This
structure can be explicitly represented in the case report form (CRF) that was
used for data collection, or it could arise because of the clinical definition
of diseases. Structured vocabularies, like the International Classification of
Diseases, define terms of increasing specificity as one moves away from the
broader terms.

Impact on variable coding
--------------------------

In CohortManager, the hierarchy is used when encoding variables for analysis or
when computing summary statistics.

.. important::

    When data is accessed any individual marked unaffected for a parent
    variable will be set to:

        - 0 if the variable is discrete
        - NaN otherwise

The rationale behind this is that individuals that are unaffected for a parent
are unaffected for all the children. Hence, any continuous or factor variable
should not be defined for this individual. This is very practical to
represent data from case report forms that often skip questions when an
individual is unaffected for a condition.

Here is a concrete example of how variable coding works in CohortManager.
First, consider the following excerpt from a CRF:

.. code::

    1. Do you have a history of arrhythmia? (Yes/No)
       - If you answered Yes to question 1, complete the following:
            1.1 Do you have a history of atrial fibrillation? (Yes/No)
                - If you answered Yes to question 1.1, complete the following:
                    + Did you suffer from paroxysmal atrial fibrillation? (Yes/No)
                    + Did you suffer from persistent atrial fibrillation? (Yes/No)
                    + Did you suffer from chronic atrial fibrillation? (Yes/No)
            1.2 How many years ago did you suffer from atrial fibrillation?

Given the previous CRF, some medical coding implementations will represent all
the skipped fields as missing data. This is incorrect, because data is not
really missing and individuals should be recognized as controls. In
CohortManager, if the parent variable is set correctly, samples will be
dynamically reclassified to account for this.

.. figure:: _static/images/hierarchy.svg
    :align: center
    :width: 100%
    :alt: Example of structured variables

There are a few things to note about this example:

1. The :py:meth:`cohort_manager.core.CohortManager.get_number_missing` is aware
   of the hierarchy and will understand that missing values are not truly
   missing if an individual is unaffected for a parent discrete variable.
2. When using :py:meth:`cohort_manager.core.CohortManager.add_data`, data
   for individuals unaffected for a parent will be ignored. The data **is not
   lost**, but it is dynamically remapped (to 0 or to NA) when accessed.
   Removing the parent is enough to get the initial interpretation.
