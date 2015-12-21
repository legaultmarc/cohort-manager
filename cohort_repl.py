#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
A REPL to access cohort information.
"""

import os
import sys
import json
import shlex
import sqlite3
import readline
import argparse

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from six.moves import input
try:
    from termcolor import colored
    COLOR = True
except ImportError:
    COLOR = False

from cohort_manager.parser import parse_yaml
from cohort_manager.core import CohortManager, CohortDataError


plt.style.use("ggplot")


REGISTERED_COMMANDS = {}
STATE = {"DEBUG": False, "PAGER": True}


def dispatch_command(line):
    if not line:
        return

    line = shlex.split(line.strip())

    # Find command.
    cmd = REGISTERED_COMMANDS.get(line[0])
    if not cmd:
        raise REPLException("Could not find command '{}'. Use 'help' to view "
                            "available commands.".format(line[0]))

    if not cmd.args_types:
        return cmd()

    # Parse the arguments.
    line = line[1:]
    n = len(line)
    # If less than minimum or more than maximum number of expected args.
    if not ((len(cmd.args_types) - cmd.optional) <= n <= len(cmd.args_types)):
        raise REPLException(
            "Command '{}' expected {} arguments ({} provided).".format(
                cmd.__name__, len(cmd.args_types), n
            )
        )

    for i, arg in enumerate(line):
        line[i] = cmd.args_types[i](arg)

    cmd(*line)


class REPLException(Exception):
    def __init__(self, message):
        self.message = message


class command(object):
    def __init__(self, *args, **kwargs):
        if args:
            # This function takes no arguments.
            f = args[0]
            f.args_types = None
            REGISTERED_COMMANDS[f.__name__] = f

        self.args_types = kwargs.get("args_types")
        self.optional = kwargs.get("optional", 0)

    def __call__(self, f):
        REGISTERED_COMMANDS[f.__name__] = f
        f.args_types = self.args_types
        f.optional = self.optional
        return f


def main():
    while True:
        try:
            if COLOR:
                s = colored("cohort repl", "blue")
            else:
                s = "cohort repl"

            line = input("[{}]> ".format(s))

            try:
                dispatch_command(line)
            except REPLException as e:
                sys.stderr.write(e.message + "\n")

        except KeyboardInterrupt:
            print()
        except EOFError:
            print()
            break
        except CohortDataError as e:
            message = "\nData integrity error.\n"
            if COLOR:
                message = colored(message, "red")
            print(message)
            print(e.value)
        except Exception as e:
            if STATE["DEBUG"]:
                # FIXME print full traceback.
                raise e

            message = "\nUnknown error occured.\n"
            if COLOR:
                message = colored(message, "red")
            print("{}\n{}".format(message, e))


def _get_manager():
    manager = STATE.get("manager")
    if not manager:
        raise REPLException("You need to load a cohort before you can "
                            "execute operations on it.")
    return manager


def _get_data_meta(phenotype):
    manager = _get_manager()
    try:
        data = manager.get_data(phenotype, numpy=True)
    except KeyError:
        raise REPLException("Could not find data for '{}'.".format(phenotype))

    meta = manager.get_phenotype(phenotype)
    if not meta:
        raise REPLException(
            "Could not find database entry for '{}'.".format(phenotype)
        )

    return data, meta


@command
def exit():
    """Quit the REPL."""
    try:
        STATE["manager"].close()
    except Exception:
        pass
    quit()


@command(args_types=(str, ), optional=1)
def help(command=None):
    """Display help about commands or list available commands."""
    if command:
        cmd = REGISTERED_COMMANDS.get(command)
        if not cmd:
            raise REPLException(
                "Can't show help for unknown command: '{}'".format(command)
            )
        print(cmd.__doc__)
    else:
        print("Available commands are:")
        for cmd in REGISTERED_COMMANDS:
            f = REGISTERED_COMMANDS[cmd]
            if COLOR:
                cmd = colored(cmd, "red")

            print("\t{}:\t{}".format(cmd, f.__doc__))


@command(args_types=(str, ))
def build(yaml_filename):
    """Build and load a cohort using a YAML descriptor."""
    STATE["manager"] = parse_yaml(yaml_filename)
    STATE["manager"].validate()


@command(args_types=(str, ))
def sql(sql):
    """Execute a SQL query on the cohort manager."""
    manager = _get_manager()
    try:
        manager.cur.execute(sql)
    except sqlite3.OperationalError as e:
        raise REPLException("Invalid SQL statement:\n{}".format(e))

    for tu in manager.cur.fetchall():
        print(tu)


@command
def list():
    """List available phenotypes."""
    manager = _get_manager()
    pager = (manager.get_number_phenotypes() > 24) and STATE["PAGER"]
    manager.tree.pretty_print(pager)


@command(args_types=(str, ))
def info(phenotype):
    """Get information and summary statistics on the phenotype."""
    data, meta = _get_data_meta(phenotype)

    print("Phenotype meta data:")
    for k, v in meta.items():
        if COLOR:
            k = colored(k, "green")
        print("\t{}{}".format(k.ljust(30), v))

    print("\nSummary statistics:")

    n_missing = STATE["manager"].get_number_missing(phenotype)
    n_total = data.shape[0]

    print("\t{} / {} missing values ({:.3f}%)".format(
        n_missing, n_total, n_missing / n_total * 100
    ))

    if meta["variable_type"] == "discrete":
        # Show information on prevalence.
        n_cases = np.sum(data == 1)
        n_controls = np.sum(data == 0)
        print("\t{} cases, {} controls; prevalence: {:.3f}%".format(
            n_cases, n_controls, n_cases / (n_cases + n_controls) * 100
        ))

    elif meta["variable_type"] == "continuous":
        mean = np.nanmean(data)
        std = np.nanstd(data)
        print(u"\tµ = {}, σ = {}".format(mean, std))
        print("\tmin = {}, max = {}".format(np.nanmin(data), np.nanmax(data)))

    elif meta["variable_type"] == "factor":
        print("\nCounts (rate):")
        n = data.shape[0]
        for name, count in data.value_counts().iteritems():
            print("\t{}: {} ({:.3f}%)".format(name, count, count / n * 100))


@command(args_types=(str, ))
def boxplot(phenotype):
    """Draw a boxplot for the given continuous phenotype."""
    data, meta = _get_data_meta(phenotype)
    data = data[~np.isnan(data)]
    if meta["variable_type"] != "continuous":
        raise REPLException("Can't draw boxplot for non-continuous phenotype "
                            "('{}').".format(phenotype))

    fig, ax = plt.subplots(1, 1)
    ax.boxplot(data, vert=False)
    ax.set_xlabel(phenotype)
    ax.set_yticklabels([])
    ax.yaxis.set_ticks_position("none")

    plt.show()


@command(args_types=(str, int), optional=1)
def histogram(phenotype, nbins=None):
    """Draw a histogram (or a bar plot for discrete variables) of the data.
    There is an optional argument for the number of bins.
    """
    data, meta = _get_data_meta(phenotype)
    data = data[~np.isnan(data)]
    if meta["variable_type"] == "continuous":
        # Histogram.
        if nbins:
            plt.hist(data, bins=nbins)
        else:
            plt.hist(data)

        plt.xlabel(phenotype)

    elif meta["variable_type"] == "discrete":
        # Bar plot.
        plt.bar((0.1, 0.4), (np.sum(data == 0), np.sum(data == 1)), width=0.1)
        plt.xticks((0.15, 0.45), ("control", "case"))
        plt.xlim((0, 0.6))

    plt.show()


@command(args_types=(str, ))
def normal_qq_plot(phenotype):
    """Plot the Normal QQ plot of the observations."""
    data, meta = _get_data_meta(phenotype)
    data = data[~np.isnan(data)]
    if meta["variable_type"] != "continuous":
        raise REPLException(
            "Could not create QQ plot for {} variable '{}'.".format(
                meta["variable_type"], phenotype
            )
        )

    data = np.sort(data)
    expected = scipy.stats.norm.ppf(
        np.arange(1, data.shape[0] + 1) / (data.shape[0] + 1),
        loc=np.mean(data),
        scale=np.std(data)
    )

    plt.scatter(expected, data, color="black", marker="o", s=10)

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        expected, data
    )
    plt.plot(
        [x_min, x_max],
        [slope * x_min + intercept, slope * x_max + intercept],
        "--", color="black",
        label="$R^2 = {:.4f}$".format(r_value ** 2)
    )
    plt.legend(loc="lower right")

    plt.xlabel("Expected quantiles")
    plt.ylabel("Observed quantiles")

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.show()


@command(args_types=(str, ))
def load(path):
    """Load the cohort from it's path on disk."""
    if not os.path.isdir(path):
        raise REPLException(
            "Could not find cohort directory '{}'.".format(path)
        )

    path = path.rstrip("/\\")
    name = path.split(os.sep)[-1]
    base = path[:-len(name)]

    if base:
        os.chdir(base)

    STATE["manager"] = CohortManager(name)
    STATE["manager"].rebuild_tree()


@command(args_types=(str, ))
def update(phenotype):
    """Update the metadata for a given phenotype."""
    data, meta = _get_data_meta(phenotype)
    meta.pop("name")

    def hook():
        readline.insert_text(json.dumps(meta))
        readline.redisplay()

    readline.set_pre_input_hook(hook)
    new_data = input("(json)>>> ")
    readline.set_pre_input_hook()

    new_data = json.loads(new_data)
    meta.update(new_data)

    # Update in db.
    _get_manager().update_phenotype(phenotype, **meta)


@command
def validate():
    """Run data balidation routine."""
    _get_manager().validate(mode="warn")


@command(args_types=(str, str, str))
def virtual(name, variable_type, expression):
    """Create a virtual variable with the given name."""
    if variable_type not in ("discrete", "continuous", "factor"):
        raise REPLException("Invalid variable type.")

    manager = _get_manager()
    try:
        variable = eval(expression, {}, dict(v=manager.variable))
    except Exception as e:
        raise REPLException("Invalid expression for virtual variable.\n" +
                            str(getattr(e, "message", str(e))))

    manager.add_phenotype(name=name, variable_type=variable_type)
    try:
        manager.add_data(name, variable.data)
    except ValueError as e:
        raise REPLException(
            "Provided data type is incorrect.\n{}".format(e.message)
        )
    manager.commit()


@command(args_types=(str, ))
def delete(phenotype):
    manager = _get_manager()
    manager.delete(phenotype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--disable-pager", action="store_false")
    args = parser.parse_args()

    STATE["DEBUG"] = args.debug
    STATE["PAGER"] = args.disable_pager
    main()
