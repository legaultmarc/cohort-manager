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
import struct
import sqlite3
import readline
import threading
import subprocess
import traceback
import argparse
import select
import socket
import logging
import webbrowser
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from six.moves import input
try:
    from termcolor import colored
    COLOR = True
except ImportError:
    COLOR = False

from cohort_manager.parser import parse_yaml
from cohort_manager.core import CohortManager
from cohort_manager.drugs.chembl import ChEMBL
from cohort_manager.drugs.drug_search import find_drugs_in_query


plt.style.use("ggplot")


logger = logging.getLogger(__name__)


REGISTERED_COMMANDS = {}
STATE = {"DEBUG": False, "PAGER": True, "PREFERRED_PORT": None}


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

    return cmd(*line)


class REPLException(Exception):
    def __init__(self, message):
        self.message = message


class command(object):
    def __init__(self, *args, **kwargs):
        if args:
            # This function takes no arguments.
            f = args[0]
            f.args_types = None
            f.printer = DefaultPrinter()
            REGISTERED_COMMANDS[f.__name__] = f

        self.args_types = kwargs.get("args_types")
        self.optional = kwargs.get("optional", 0)
        self.printer = kwargs.get("printer")

    def __call__(self, f):
        REGISTERED_COMMANDS[f.__name__] = f
        f.args_types = self.args_types
        f.optional = self.optional
        f.printer = self.printer
        return f


class DefaultPrinter(object):
    def __call__(self, res):
        output = StringIO()

        res = res.decode("utf-8")
        try:
            res = json.loads(res)
        except Exception:
            print(res, file=output)
            return

        if type(res) is dict:
            message = res.get("message", "")
        else:
            message = False

        if message:
            print(message, file=output)
        else:
            print(json.dumps(res, indent=4), file=output)

        n_lines = len(output.getvalue().splitlines())
        if n_lines > 20:
            self._pager_print(output.getvalue())
        else:
            print(output.getvalue())

    def _pager_print(self, s):
        try:
            proc = subprocess.Popen(["less", ], stdin=subprocess.PIPE)
        except Exception:
            print(s)
            return

        proc.communicate(input=s.encode("utf-8"))


class ImagePrinter(DefaultPrinter):
    def __call__(self, res):
        res = json.loads(res.decode("utf-8"))
        print("Opening generated image ('{}').".format(res["image_path"]))
        webbrowser.open("file://" + res["image_path"])


def _server(stop_event, headless, sock):
    log = STATE["DEBUG"] or headless
    try:
        # Wait for a connection.
        while True:
            if stop_event.is_set():
                # We were asked to quit by the main thread.
                raise SystemExit

            try:
                readable, writeable, errored = select.select(
                    [sock], [], [], 1.0
                )
            except OSError:
                # Socket was closed by the main thread and we should terminate.
                raise SystemExit

            if readable:
                break

        # A client showed up so we accept the connection.
        conn, addr = sock.accept()
        if log:
            print("Client connected! ({}).".format(addr))

        while not stop_event.is_set():
            conn.settimeout(1.0)
            try:
                # Get a command from the client.
                command = conn.recv(4096)
            except socket.timeout:
                # This will happen every second.
                continue
            except ConnectionResetError:
                # Client closed connection.
                # TODO Prepare to handle new connection.
                break

            if not command or command == b"close\n":
                break

            # We process the received command.
            res = _handle_command(command)
            message = _build_msg(
                json.dumps(res, indent=4).encode("utf-8")
            )

            # Make this blocking.
            conn.settimeout(None)
            conn.sendall(message)

    finally:
        if log:
            print("Server closing.")
        try:
            conn.close()
        except Exception:
            pass


def _handle_command(raw_command):
    try:
        command = raw_command.decode("utf-8").strip()
        res = dispatch_command(command)
    except Exception as e:
        error_message = (
            "\n{} was raised when handling command.\n"
            "Traceback:\n{}"
        )
        error_message = error_message.format(
            type(e).__name__,
            "".join(traceback.format_tb(sys.exc_info()[2]))
        )

        logger.warning(error_message)

        res = {"success": False}
        if hasattr(e, "message"):
            res["message"] = e.message
        elif hasattr(e, "value"):
            res["message"] = e.value
        elif hasattr(e, "args"):
            if e.args:
                res["message"] = e.args[0]

    if res is None:
        res = {"succes": False, "message": "Empty response."}

    return res


def main(headless):
    # Set up the server socket.
    stop_event = threading.Event()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if STATE["PREFERRED_PORT"] is not None:
        sock.bind(("", STATE["PREFERRED_PORT"]))
    else:
        sock.bind(("", 0))

    port = sock.getsockname()[1]
    sock.listen(1)

    if STATE["DEBUG"] or headless:
        print("Server listening on port {}.".format(port))

    # Launch the server thread.
    server = threading.Thread(target=_server, daemon=False,
                              args=(stop_event, headless, sock))
    server.start()

    try:
        if headless:
            # In headless mode, the only thing that the main thread will do is
            # to wait for interrupt.
            try:
                while server.is_alive():
                    pass
            except:
                stop_event.set()
                return

        else:
            client(port)
            server.join()

    finally:
        sock.close()


def _build_msg(s):
    """Build a message by prefixing it's length."""
    # > means big endiang and I means unsigned int.
    # This will use 4 bytes (see standard size in the struct docs).
    return struct.pack(">I", len(s)) + s


def _read_msg(sock):
    """Read a message from an open socket."""
    # Read the 4 bytes of message size.
    message_length = sock.recv(4)
    message_length = struct.unpack(">I", message_length)[0]

    data = b""
    while len(data) < message_length:
        packet = sock.recv(4096)
        if not packet:
            logger.warning("Truncated message.")
            return None
        data += packet
    return data


def client(port):
    # Built-in client for non-headless connections.
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("", port))

    while True:
        try:
            if COLOR:
                s = colored("cohort repl", "blue")
            else:
                s = "cohort repl"

            cmd = input("[{}]> ".format(s))
            if not cmd:
                continue
            if cmd == "exit":
                raise EOFError()

            # Send the command to the server.
            client_sock.sendall(cmd.encode("utf-8"))

            # Get the results printer.
            cmd_name = shlex.split(cmd.strip())
            try:
                printer = REGISTERED_COMMANDS[cmd_name[0]].printer()
            except Exception:
                printer = DefaultPrinter()

            # Get the response from the server.
            res = _read_msg(client_sock)

            printer(res)

        except KeyboardInterrupt:
            print()

        except EOFError:
            print("Bye")
            client_sock.sendall(b"close\n")
            client_sock.close()
            break


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
    """Quit the REPL.

    This is only available when it is not running in headless mode.

    """
    pass  # Behaviour implemented directly in main()


@command(args_types=(str, ), optional=1)
def help(command=None):
    """Display help about commands or list available commands.

    :param command: The name of the command to get it's specific help message.
    :type command: str

    If nor argument is provided, the short help message will be printed for all
    available commands. If the provided command name is not found, an exception
    will be raised.

    """
    message = StringIO()
    if command:
        cmd = REGISTERED_COMMANDS.get(command)
        if not cmd:
            raise REPLException(
                "Can't show help for unknown command: '{}'".format(command)
            )
        print(_format_long_doc(cmd), file=message)
    else:
        print("Available commands are:", file=message)
        longest_cmd = len(max(REGISTERED_COMMANDS, key=lambda x: len(x))) + 10
        for cmd in REGISTERED_COMMANDS:
            f = REGISTERED_COMMANDS[cmd]
            print("  {}:{space}{}".format(
                cmd,
                _get_short_doc(f),
                space=" " * (longest_cmd - len(cmd))
            ), file=message)

    return {"success": True, "message": message.getvalue()}


@command(args_types=(str, ))
def build(yaml_filename):
    """Build and load a cohort using a YAML descriptor.

    :param yaml_filename: The filename of the YAML cohort descriptor.
    :type yaml_filename: str

    This command will build the cohort and load it in the current session.

    """
    STATE["manager"] = parse_yaml(yaml_filename)
    STATE["manager"].validate()

    return {"success": True, "message":
            "Loaded cohort {}.".format(yaml_filename)}


@command(args_types=(str, ))
def sql(sql):
    """Execute a SQL query on the cohort manager.

    :param sql: The SQL query.
    :type sql: str

    This can be used to directly interrogate the relational database
    underlying the cohort manager.

    """
    manager = _get_manager()
    try:
        manager.cur.execute(sql)
    except sqlite3.OperationalError as e:
        raise REPLException("Invalid SQL statement:\n{}".format(e))

    return {"success": True, "results": manager.cur.fetchall()}


@command
def list():
    """List available phenotypes.

    This will display all the available phenotypes for the currently loaded
    cohort. A pager (e.g. less) will automatically be used if the number
    of variables is greater than 24.

    """
    manager = _get_manager()
    message = StringIO()
    _stdout = sys.stdout
    sys.stdout = message
    manager.tree.pretty_print()
    sys.stdout = _stdout
    return {"success": True, "message": message.getvalue()}


@command(args_types=(str, ))
def info(phenotype):
    """Get information and summary statistics on a phenotype.

    :param phenotype: The name of the phenotype to get summary information on.
    :type phenotype: str

    Use 'list' to see all the available phenotypes for this command.

    """
    message = StringIO()
    data, meta = _get_data_meta(phenotype)

    print("Phenotype meta data:", file=message)
    for k, v in meta.items():
        if COLOR:
            k = colored(k, "green")
        print("\t{}{}".format(k.ljust(30), v), file=message)

    print("\nSummary statistics:", file=message)

    n_missing = STATE["manager"].get_number_missing(phenotype)
    n_total = data.shape[0]

    print("\t{} / {} missing values ({:.3f}%)".format(
        n_missing, n_total, n_missing / n_total * 100
    ), file=message)

    if meta["variable_type"] == "discrete":
        # Show information on prevalence.
        n_cases = np.sum(data == 1)
        n_controls = np.sum(data == 0)
        print("\t{} cases, {} controls; prevalence: {:.3f}%".format(
            n_cases, n_controls, n_cases / (n_cases + n_controls) * 100
        ), file=message)

    elif meta["variable_type"] == "continuous":
        mean = np.nanmean(data)
        std = np.nanstd(data)
        print(u"\tµ = {}, σ = {}".format(mean, std), file=message)
        print("\tmin = {}, max = {}".format(np.nanmin(data), np.nanmax(data)),
              file=message)

    elif meta["variable_type"] == "factor":
        print("\nCounts (rate):", file=message)
        n = data.shape[0]
        for name, count in data.value_counts().iteritems():
            print("\t{}: {} ({:.3f}%)".format(name, count, count / n * 100),
                  file=message)

    return {"success": True, "message": message.getvalue()}


@command(args_types=(str, ), printer=ImagePrinter)
def boxplot(phenotype):
    """Draw a boxplot for the given continuous phenotype.

    :param phenotype: Display a boxplot for the provided continuous phenotype.
    :type phenotype: str

    An exception will be raised if the required phenotype is not continuous.

    """
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

    filename = "cohort_plot.png"
    plt.savefig(filename, dpi=300)
    return _response_from_img_filename(filename)


def _response_from_img_filename(path):
    plt.clf()
    return {
        "success": True,
        "image_path": os.path.join(
            os.path.abspath(""), path
        )
    }


@command(args_types=(str, str), printer=ImagePrinter)
def scatter(y, x):
    """Plot two continuous phenotypes in a scatterplot.

    :param y: The phenotype to plot as the y axis.
    :type y: str

    :param x: The phenotype to ploy as the x axis.
    :type x: str

    This function is only available for continuous phenotypes.

    """
    datay, metay = _get_data_meta(y)
    datax, metax = _get_data_meta(x)

    if not (metax["variable_type"] == "continuous" and
            metay["variable_type"] == "continuous"):
        raise REPLException("Can only plot scatter for continuous variables.")

    not_missing = ~ (np.isnan(datax) | np.isnan(datay))

    plt.plot(datay[not_missing], datax[not_missing], ".", c="black", ms=2)
    plt.xlabel(x)
    plt.ylabel(y)
    filename = "cohort_plot.png"
    plt.savefig(filename, dpi=300)

    return _response_from_img_filename(filename)


@command(args_types=(str, int), optional=1, printer=ImagePrinter)
def histogram(phenotype, nbins=None):
    """Draw a histogram (or a bar plot for discrete variables) of the data.

    :param phenotype: The phenotype for which to draw the histogram.
    :type phenotype: str

    :param nbins: The number of bins for the histogram (optional).
    :type nbins: int

    This function will work on both continuous and discrete variables (but
    not factors).

    """
    data, meta = _get_data_meta(phenotype)
    if meta["variable_type"] not in ("continuous", "discrete"):
        raise REPLException("Could not generate histogram for '{}' variable."
                            "".format(meta["variable_type"]))

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

    filename = "cohort_plot.png"
    plt.savefig(filename, dpi=300)

    return _response_from_img_filename(filename)


@command(args_types=(str, ), printer=ImagePrinter)
def normal_qq_plot(phenotype):
    """Plot the Normal QQ plot of the observations.

    :param phenotype: The phenotype for which to draw the QQ plot.
    :type phenotype: str

    This function is only available for continuous phenotypes.

    """
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

    filename = "cohort_plot.png"
    return _response_from_img_filename(filename)


@command(args_types=(str, ))
def load(path):
    """Load the cohort from it's path on disk.

    :param path: The path to the cohort directory.
    :type path: str

    This will bind a manager instance to the REPL and will initialize the
    phenotype tree.

    """
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
    return {"success": True, "message": "Loaded cohort at '{}'.".format(path)}


@command(args_types=(str, ))
def update(phenotype):
    """Update the metadata for a given phenotype.

    :param phenotype: The phenotype for which the metadata will be updated.
    :type phenotype: str

    This will display a JSON string that the user can edit to update the
    database.

    Note that it is not possible to rename phenotypes using this function as
    it is used as the database key to access the data.

    TODO. Make this better for the new (socket) REPL.

    """
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

    return {"success": True, "message": "Phenotype updated in database."}


@command
def validate():
    """Run data validation routine."""
    _get_manager().validate(mode="warn")
    return {
        "success": True,
        "message": ("Ran validation routine (warnings have been displayed if "
                    "necessary.)")
    }


@command(args_types=(str, str, str))
def virtual(name, variable_type, expression):
    """Create a virtual variable.

    :param name: The name of the new variable/phenotype.
    :type name: str

    :param variable_type: The type of the new variable. This should be
                          consistent with the provided expression.
    :type variable_type: str

    :param expression: A valid Python expression that will be used to create
                       the virtual variable (see examples).
    :type expression: str

    Examples:

        # Create a new continuous variable that is the log transform of another.
        [cohort repl]> virtual logWeight continuous 'v("Weight").log()'

        # Create a new discrete variable.
        [cohort repl]> virtual myVar discrete '(v("Age") > 50) & v("MyocardialInfarction")'

        # Compute z-scores.
        [cohort repl]> virtual z continuous '(v("Weight") - v("Weight").mean()) / v("Weight").std()'

    Note that the `v()` function is used to allow users to create virtual
    variables based on currently existing phenotypes. Continuous phenotypes
    have special methods like 'log', 'mean', 'std' and support operators like
    '-', '+', '/', '*', '**' as well as comparators '>', '<', '<=', '>=', '==',
    '!='.

    Discrete variables support most of the boolean operators '&', '|' and
    can be inverted using the '~' operator.

    This procedure correctly handles missing values for virtual variables.

    For Python programmers, note that `v()` is an alias to
    `CohortManager.variable()` which returns a `cohort_manager.core._Variable`
    instance.

    """
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

    return {
        "success": True,
        "message": ("Successfully created a new {} virtual variable named {}."
                    "".format(variable_type, name))
    }


@command(args_types=(str, ))
def delete(phenotype):
    """Delete the given phenotype.

    :param phenotype: The name of the phenotype.
    :type phenotype: str

    This function will not try to stop you and it can't be reverted.

    """
    manager = _get_manager()
    manager.delete(phenotype)
    return {
        "success": True,
        "message": "Deleted phenotype '{}' from database.".format(phenotype)
    }


@command(args_types=(int, ))
def drug_info(molregno):
    """Get information about a drug given it's ChEMBL molregno identifier.

    :param molregno: The ChEMBL molregno, a molecule identifier.
    :type molregno: int

    The molregno can be found using the 'drug_search' function.
    Note that this function only works if ChEMBL is installed locally and
    if the configuration is valid. For more information on setting up ChEMBL,
    refer to the documentation on `cohort_manager.drugs.chembl.ChEMBL`.

    """
    with ChEMBL() as db:
        return db.get_drug_info(molregno)


@command(args_types=(str, float), optional=1)
def drug_search(s, min_score=None):
    """Query ChEMBL to find a drug corresponding to the provided query.

    :param s: The query string (a drug name).
    :type s: str

    :param min_score: The minimum similarity score (between 0 and 1) to report
                      the results (Optional).
    :type min_score: float

    """
    if min_score is not None:
        results = find_drugs_in_query(s, min_score)
    else:
        results = find_drugs_in_query(s)

    results = sorted(results, key=lambda x: x[-1], reverse=True)

    fields = ("molregno", "matching_name", "score")
    results = [dict(zip(fields, i)) for i in results]

    return {
        "success": True,
        "hits": results
    }


def _get_short_doc(f):
    """Returns the first line of the doctstring."""
    s = f.__doc__
    if not s:
        return ""

    s = s.splitlines()
    if len(s) == 1:
        return s[0]

    # Find the first empty line.
    idx = -1
    for idx, chunk in enumerate(s):
        if chunk.isspace() or chunk == "":
            break

    if idx != -1:
        return " ".join(
            [i.strip() for i in s[:idx] if not i.isspace() and i != ""]
        )

    return s[0]


def _format_long_doc(f):
    """Formats a docstring to print it to screen."""
    if not f.__doc__:
        return ""

    li = f.__doc__.splitlines()
    s = ""
    done_first_line = False
    for chunk in li:
        if not done_first_line and (chunk.isspace() or chunk == ""):
            done_first_line = True
            s += "\n\n"

        elif not done_first_line:
            s += " {}".format(chunk.strip())

        else:
            s += "{}\n".format(chunk)

    return s


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--disable-pager", action="store_false")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--port", default=None, type=int)
    args = parser.parse_args()

    STATE["DEBUG"] = args.debug
    STATE["PAGER"] = args.disable_pager
    STATE["PREFERRED_PORT"] = args.port
    main(headless=args.headless)


if __name__ == "__main__":
    entry_point()
