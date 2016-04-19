"""
Utilities to manage ATC codes.
"""

import re


def get_atc_code_level(atc_code):
    """Get the level of an ATC code.

    :param atc_code: The ATC code.
    :type atc_code: str

    :returns: The level of the ATC code (1, 2, 3, 4 or 5).
    :rtype: int

    This function will raise a ValueError if the ATC code is invalid.

    """
    levels = {
        "^[ABCDGHJLMNPRSV]$": 1,
        "^[ABCDGHJLMNPRSV][0-9]{2}$": 2,
        "^[ABCDGHJLMNPRSV][0-9]{2}[A-Z]$": 3,
        "^[ABCDGHJLMNPRSV][0-9]{2}[A-Z]{2}$": 4,
        "^[ABCDGHJLMNPRSV][0-9]{2}[A-Z]{2}[0-9]{2}$": 5,
    }

    matched_level = None
    for regex, level in levels.items():
        if re.match(regex, atc_code.upper()):
            matched_level = level

    if not matched_level:
        raise ValueError("Could not parse ATC code '{}'.".format(atc_code))

    return matched_level
