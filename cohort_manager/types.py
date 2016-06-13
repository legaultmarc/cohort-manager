"""
Module to organize variable types.
"""

import logging

import numpy as np

from . import inference


logger = logging.getLogger(__name__)


class InvalidValues(Exception):
    def __init__(self, message):
        self.message = message


def _build_types_dict():
    types = set()
    stack = set()

    stack.add(Type)

    while stack:
        cur = stack.pop()
        types.add(cur)

        # Add it's subclasses
        for cls in cur.__subclasses__():
            stack.add(cls)

    return {cls.__name__.lower(): cls for cls in types}


def type_str(s):
    """Return a type class from its name."""
    try:
        return TYPES_DICT[s.lower()]
    except KeyError:
        raise ValueError("Unknown variable type '{}'.".format(s))


def is_type_name(s):
    """Checks if a string is the name of a type."""
    return not (TYPES_DICT.get(s.lower()) is None)


class Type(object):
    """Parent type for all types."""
    @staticmethod
    def check(values):
        """Check if a data vector is of the appropriate type.

        Raises an InvalidValues exception as a way of conveniently passing the
        error message if relevant.
        """
        raise NotImplementedError()

    @classmethod
    def subtype_of(cls, parent):
        return issubclass(cls, parent)


# Type subclasses.
class Discrete(Type):
    @staticmethod
    def check(values):
        try:
            values = np.array(values, dtype=float)
        except ValueError:
            raise InvalidValues("Some values are non-numeric for Discrete "
                                "variable.")

        extra = set(np.unique(values[~np.isnan(values)])) - {0, 1}
        if len(extra) != 0:
            extra = ", ".join([str(i) for i in list(extra)[:5]])
            if len(extra) > 5:
                extra += ", ..."
            raise InvalidValues(
                "Authorized values for discrete variables are 0, 1 and np.nan."
                "\nUnexpected values were observed ({})."
                "".format(extra)
            )

        return True


class Continuous(Type):
    @staticmethod
    def check(values, _raise=False):
        """Check that the data vector is continuous.

        This is very hard to check.

        After taking the outlier data points (arbitrarily defined as):

            |x| > 3 * median absolute deviation + median

        We take the most common outlier if it consists of at least 50% of the
        identified data points. If such a value exist, we suggest to the
        user that it might be a weird value arising from bad missing variable
        coding.

        Also, if there is less than 5 distinct values in either the full vector
        or a random sample of 5000 elements, a warning will be displayed
        prompting the user to verify the data type.

        """
        try:
            values = np.array(values, dtype=np.float)
        except ValueError:
            raise InvalidValues("Non-numeric values in Continuous variable.")

        values = values[~np.isnan(values)]

        # Factor check.
        n_values = inference.estimate_num_distinct(values)

        if n_values < 5:
            message = ("There is a lot of redundancy in the values of this "
                       "continuous variable. Perhaps it should be modeled "
                       "as a factor or another variable type.")
            if _raise:
                raise InvalidValues(message)
            logger.warning(message)  # pragma: no cover

        # Outlier check.
        common_outlier = inference.find_overrepresented_outlier(values)
        if common_outlier is not None:
            message = ("The value '{}' is commonly found in the tails of the "
                       "distribution. This could be because of bad coding of "
                       "missing values.".format(common_outlier))
            if _raise:
                raise InvalidValues(message)
            logger.warning(message)  # pragma: no cover


class PositiveInteger(Continuous):
    @staticmethod
    def check(values):
        valid = True
        message = ""
        try:
            data = np.array(values, dtype=np.float)
        except Exception:
            valid = False
            message = ("Some of the provided values are non-numeric.")

        if valid:
            if np.any(data < 0):
                valid = False
                message = ("Negative values were observed while the type is "
                           "PositiveInteger.")

            elif not np.can_cast(data, np.int):
                valid = False
                message = "Some of the provided values are not integers."

        if not valid:
            raise InvalidValues(message)
        return valid


class Factor(Type):
    @staticmethod
    def check(values, mapping):
        """Check that the data vector is consistent with a factor variable.

        This is done by looking at the code from the database and making sure
        that all the observed integer codes are defined.

        """
        # Check if the set of observed values is consistent with the code.
        observed = set(values[~np.isnan(values)])
        expected = set([i[0] for i in mapping])
        extra = observed - expected
        if extra:
            raise InvalidValues("Unknown encoding value(s) ({}) for factor "
                                "variable.".format(extra))


class Date(Type):
    pass


class PastDate(Date):
    pass


TYPES_DICT = _build_types_dict()
