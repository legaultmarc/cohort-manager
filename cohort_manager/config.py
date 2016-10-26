"""
Module to manage configuration files for the CohortManager.
"""


import configparser
import logging
import os


logger = logging.getLogger(__name__)


class Config(object):
    def __init__(self):
        self.reset()

    def reset(self):
        # ChEMBL defaults.
        self.chembl = {
            "name": "chembl_21",
            "host": "localhost",
            "username": "",
            "port": "5432",
            "b64_password": "",
            "password": "",
        }

        # SNOMED-CT defaults.
        self.snomed_ct = {
            "name": "snomed",
            "host": "localhost",
            "username": "",
            "b64_password": "",
            "password": "",
        }

        # CohortManager Backend.
        self.backend = {
            "name": "SQLiteBackend",
        }

    def write(self, filename):
        config = configparser.ConfigParser()
        for k, v in self.__dict__.items():
            config[k.upper()] = v

        with open(filename, "w") as f:
            config.write(f)

    def read(self, filename):
        self.reset()

        config = configparser.ConfigParser()
        config.read(filename)

        for section in config.sections():
            for key in config[section]:
                s = self.__dict__.get(section.lower())
                if s is None:
                    raise ValueError("Invalid section '{}'.".format(section))

                if key not in s:
                    raise ValueError(
                        "Invalid parameter '{}' in section '{}'."
                        "".format(key, section)
                    )

                s[key] = config[section][key]


def detect_config():
    cfg = Config()
    filename = "cohort_managerrc.ini"

    # Create config directory if necessary.
    config_dir = os.path.join(
        os.path.expanduser("~"),
        ".cohort_manager"
    )
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)

    filenames = [
        filename,
        os.path.join(config_dir, filename)
    ]

    for filename in filenames:
        if os.path.isfile(filename):
            cfg.read(filename)
            return cfg

    # Need to create the default file.
    logger.info(
        "Creating default configuration file in '{}'.".format(config_dir)
    )
    cfg.write(filenames[-1])
    return cfg


configuration = detect_config()
