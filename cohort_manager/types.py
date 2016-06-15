"""
Module to organize variable types.
"""

import datetime
import logging

import numpy as np
import pandas as pd

from . import stats


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


def estimate_num_distinct(v):
    """Estimate the number of distinct elements in vector."""
    try:
        v = v[~np.isnan(v)]
    except TypeError:  # Not a vector of floats.
        pass

    if v.shape[0] <= 5000:
        return len(np.unique(v))
    else:
        # len(np.unique) is ~3x faster than len(set(x))
        return len(np.unique(np.random.choice(v, 5000, replace=False)))


def generate_type_graph():
    """Generate a dot file representing the type graph."""
    with open("cohort_manager_types_graph.dot", "w") as f:
        print("digraph cohort_manager_type_graph {", file=f)
        for name, cls in TYPES_DICT.items():
            for child in cls.__subclasses__():
                print(
                    "\t{} -> {}".format(cls.__name__, child.__name__), file=f
                )
        print("}", file=f)


def _get_numpy_float_array(values):
    try:
        return np.array(values, dtype=np.float)
    except ValueError:
        pass

    raise InvalidValues("Some values are non-numeric.")


class Type(object):
    """Parent type for all types."""
    @staticmethod
    def check(values):
        """Check if a data vector is of the appropriate type.

        Raises an InvalidValues exception as a way of conveniently passing the
        error message if relevant.
        """
        raise NotImplementedError()

    @staticmethod
    def encode(values):
        """Method that gets called when storing the data in the HDF5 dataset.

        This can be overriden by children who need a special encoding step.

        .. note::

            This function gets called automatically and can be used to encode
            arrays that are not handled correctly by H5PY. It should not be
            used to process information.

        """
        return values

    @staticmethod
    def decode(hdf5_dataset):
        """Method that is used to restore data from HDF5.

        If a special encoding method was provided, implementing this method
        is essential.

        """
        return np.array(hdf5_dataset)

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
        values = _get_numpy_float_array(values)
        values = values[~np.isnan(values)]

        # Factor check.
        n_values = estimate_num_distinct(values)

        if n_values < 5:
            message = ("There is a lot of redundancy in the values of this "
                       "continuous variable. Perhaps it should be modeled "
                       "as a factor or another variable type.")
            if _raise:
                raise InvalidValues(message)
            logger.warning(message)  # pragma: no cover

        # Outlier check.
        common_outlier = stats.find_overrepresented_outlier(values)
        if common_outlier is not None:
            message = ("The value '{}' is commonly found in the tails of the "
                       "distribution. This could be because of bad coding of "
                       "missing values.".format(common_outlier))
            if _raise:
                raise InvalidValues(message)
            logger.warning(message)  # pragma: no cover


class Integer(Continuous):
    @staticmethod
    def check(values):
        super(Integer, Integer).check(values)
        data = _get_numpy_float_array(values)
        data = data[~np.isnan(data)]
        if np.nansum((data * 10) % 10) != 0:
            raise InvalidValues(
                "Some of the provided values are not integers."
            )


class PositiveInteger(Integer):
    @staticmethod
    def check(values):
        super(PositiveInteger, PositiveInteger).check(values)
        data = _get_numpy_float_array(values)
        if np.any(data < 0):
            examples = np.unique(data[np.where(data < 0)[0]])[:4]
            examples = ", ".join([str(i) for i in examples])
            raise InvalidValues("Negative values were observed while the type "
                                "is PositiveInteger (e.g. {})."
                                "".format(examples))


class Year(PositiveInteger):
    @staticmethod
    def check(values):
        super(Year, Year).check(values)


class NegativeInteger(Integer):
    @staticmethod
    def check(values):
        super(NegativeInteger, NegativeInteger).check(values)

        data = _get_numpy_float_array(values)
        if np.any(data > 0):
            examples = np.unique(data[np.where(data > 0)[0]])[:4]
            examples = ", ".join([str(i) for i in examples])
            raise InvalidValues("Positive values were observed while the type "
                                "is NegativeInteger (e.g. {})."
                                "".format(examples))


class Factor(Type):
    @staticmethod
    def check(values, mapping):
        """Check that the data vector is consistent with a factor variable.

        This is done by looking at the code from the database and making sure
        that all the observed integer codes are defined.

        """
        # Check if the set of observed values is consistent with the code.
        values = _get_numpy_float_array(values)
        observed = set(values[~np.isnan(values)])
        expected = set([i[0] for i in mapping])
        extra = observed - expected
        if extra:
            raise InvalidValues("Unknown encoding value(s) ({}) for factor "
                                "variable.".format(extra))


class Date(Type):
    @classmethod
    def check(cls, values):
        """Checks that the list of strings can be used to represent ISO 8601
        dates.

        The simple representation without the time is used (e.g. 2016-05-28).

        """
        valid = True
        for i in values:
            missing = i is None
            try:
                if np.isnan(i):
                    missing = True
            except TypeError:
                pass

            if missing:
                continue

            try:
                date = cls._parse_date(i)
            except Exception:
                valid = False

            if not valid:
                raise InvalidValues(
                    "Some of the values are not ISO 8601 dates. The expected "
                    "format is: YYYY-MM-DD and some of the provided values "
                    "had the following form: {}.".format(i)
                )

            one_day = datetime.timedelta(days=1)
            bounds = (
                pd.Timestamp.min.to_datetime().date() + one_day,
                pd.Timestamp.max.to_datetime().date() - one_day,
            )

            if not (bounds[0] < date < bounds[1]):
                raise InvalidValues(
                    "Because date are stored using Pandas, they are encoded "
                    "with nanosecond resolution and need to be between "
                    "bounds defined by {} and {}. The following date is "
                    "out of these bounds: {}.".format(*bounds, date)
                )

    @classmethod
    def predicate(cls, s):
        """Test if a string is a valid date (returns True or False)."""
        try:
            cls._parse_date(s)
            return True
        except Exception:
            return False

    @staticmethod
    def _parse_date(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()

    @staticmethod
    def date_to_int(date):
        if date is None:
            return date

        return date.year * 10000 + date.month * 100 + date.day

    @staticmethod
    def int_to_date(i):
        """Takes an int representing a date and converts it to a datetime
        object.

        The representation is simply: YYYMMDD

        """
        if i is None or np.isnan(i):
            return i

        i = int(i)
        year = i // 10000
        i -= year * 10000
        month = i // 100
        day = i - month * 100
        return datetime.date(year=year, month=month, day=day)

    @classmethod
    def encode(cls, values, from_type=str):
        if not values:
            return values

        # If items are already dates, we just convert to int.
        if from_type is datetime.date:
            return np.array([cls.date_to_int(d) for d in values])

        # Else we assume they are strings and parse the dates.
        if from_type is str:
            n = len(values)
            out = np.full(n, np.nan)
            for i in range(n):
                try:
                    date = cls._parse_date(values[i])
                except Exception:
                    date = None

                if date:
                    out[i] = cls.date_to_int(date)

            return out

    @classmethod
    def decode(cls, values):
        return pd.to_datetime(pd.Series([cls.int_to_date(i) for i in values]))


class PastDate(Date):
    @classmethod
    def check(cls, values):
        super(PastDate, PastDate).check(values)
        for i in values:
            date = cls._parse_date(i)
            if date > datetime.datetime.today().date():
                raise InvalidValues(
                    "Some of the values are future dates (e.g. {})."
                    "".format(i)
                )


TYPES_DICT = _build_types_dict()
