"""
Integration with the ModelSpec API from genetest to allow statistical testing.
"""

import argparse
import logging

logger = logging.getLogger(__name__)

import genetest.genotypes
from ..bindings.genetest import CohortManagerContainer
from genetest.modelspec import parse_modelspec, ModelSpec
from genetest.analysis import execute
from genetest.subscribers import Subscriber

from genetest.statistics.models.linear import StatsLinear
from genetest.statistics.models.logistic import StatsLogistic


def main(args):
    # Genotypes
    if args.genotypes_format and args.genotypes:
        container_class = genetest.genotypes.format_map(args.genotypes_format)
        genotypes = container_class(args.genotypes)
    else:
        logger.info("Doing analysis with no genotype information.")
        genotypes = None

    # Phenotypes
    phenotypes = CohortManagerContainer(args.cohort)

    # Model or models.
    models = None
    try:
        with open(args.model) as f:
            models = [i.strip() for i in f.readlines()]
    except FileNotFoundError:
        pass

    if models is None:
        models = [args.model]

    # Create the subscriber.
    subscriber = Print(args.output_format)

    # Create the statistical test.
    test_kwargs = {}
    if args.test_kwargs:
        test_kwargs = dict(
            [i.split("=") for i in args.test_kwargs.split(",")]
        )
        for k, v in test_kwargs.items():
            if v.startswith("float:"):
                test_kwargs[k] = float(test_kwargs[k][6:])

    def get_test_factory(name, kwargs):
        tests = {"linear": StatsLinear, "logistic": StatsLogistic}
        return lambda: tests[name](**kwargs)

    # Parse the model(s).
    for model in models:
        model = parse_modelspec(model)
        model["test"] = get_test_factory(args.test, test_kwargs)

        model = ModelSpec(**model)

        execute(phenotypes, genotypes, model, subscribers=[subscriber])


class Tracker(object):
    def __init__(self, path=None):
        if path is None:
            self.path = []
        else:
            self.path = path

    def __getitem__(self, k):
        return Tracker(self.path + [k])


class Print(Subscriber):
    def __init__(self, fmt, sep="\t"):
        self.fmt = eval(fmt, {}, {"res": Tracker()})
        assert type(self.fmt) is list

        self.SEP = sep

    def handle(self, results):
        results = Subscriber._apply_translation(
            self.modelspec.get_translations(),
            results
        )

        out = []
        for column in self.fmt:
            if isinstance(column, str):
                out.append(column)

            elif isinstance(column, Tracker):
                cur = results
                for crumb in column.path:
                    cur = cur[crumb]

                out.append(str(cur))

        print(self.SEP.join(out))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        help=("A formula of the form 'y ~ x1 + x2' or a file containing one "
              "formula per line.")
    )

    parser.add_argument(
        "--cohort",
        help="The cohort."
    )

    parser.add_argument(
        "--output-format", "--f",
        help=("The format to output the results. This string will be "
              "evaluated in an environment with a 'res' object that can "
              "be used as an accessor. For example, if there is a "
              "predictor called 'x', one can use '[res['x']['p_value']]' "
              "as a valid output format.")
    )

    parser.add_argument(
        "--genotypes", "-g",
        help="Path to the genotypes file.",
        default=None
    )

    parser.add_argument(
        "--genotypes-format",
        help="A genetest genotypes container string (e.g. plink).",
        default=None
    )

    parser.add_argument(
        "--test",
        help="A genetest statistical test name."
    )

    parser.add_argument(
        "--test-kwargs",
        help=("kwargs to pass to initialize the statistical test. The format "
              "is key=value."),
        default=""
    )

    main(parser.parse_args())
