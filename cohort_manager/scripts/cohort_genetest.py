"""
Integration with the ModelSpec API from genetest to allow statistical testing.
"""

import argparse
import json
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

import numpy as np

from genetest.genotypes.core import Representation
from genetest.genotypes import format_map
from ..bindings.genetest import CohortManagerContainer
from genetest.modelspec import parse_modelspec, ModelSpec, _reset
from genetest.analysis import execute
from genetest.subscribers import Subscriber

from genetest.statistics.core import StatsError
from genetest.statistics.models.linear import StatsLinear
from genetest.statistics.models.logistic import StatsLogistic


def main(args):
    # Genotypes
    if args.genotypes_format and args.genotypes:
        container_class = format_map[args.genotypes_format]
        genotypes = container_class(args.genotypes,
                                    representation=Representation.ADDITIVE)
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

    # Remove comments or blank lines.
    models = [i for i in models if i != "" and (not i.startswith("#"))]

    # Create the statistical test.
    test_kwargs = {}
    if args.test_kwargs:
        test_kwargs = dict(
            [i.split("=") for i in args.test_kwargs.split(",")]
        )
        for k, v in test_kwargs.items():
            if v.startswith("float:"):
                test_kwargs[k] = float(test_kwargs[k][6:])
            elif v.startswith("bool:"):
                if v == "bool:False":
                    test_kwargs[k] = False
                elif v == "bool:True":
                    test_kwargs[k] = True
                else:
                    raise ValueError("Invalid bool '{}'.".format(v[5:]))

    def get_test_factory(name, kwargs):
        tests = {"linear": StatsLinear, "logistic": StatsLogistic}
        return lambda: tests[name](**kwargs)

    # Parse the model(s).
    for model_str in models:
        model = parse_modelspec(model_str)
        model["test"] = get_test_factory(args.test, test_kwargs)

        # Conditioning aka subgroup or stratified.
        conditions = model.pop("conditions")
        if conditions is not None:
            model["stratify_by"] = [i["name"] for i in conditions]
            subgroups = [i["level"] for i in conditions]
        else:
            subgroups = None

        model = ModelSpec(**model)

        try:
            execute(phenotypes, genotypes, model,
                    subscribers=[Print(model_str)], subgroups=subgroups)
        except StatsError as e:
            logger.warning(
                "Exception raised while fitting:\n{}\n"
                "Exception message: {}".format(
                    model_str, str(e)
                )
            )
            if not args.log_errors:
                raise e

        _reset()


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "dtype"):
            if np.issubdtype(o.dtype, int):
                o = o.item()
            elif np.issubdtype(o.dtype, float):
                o = o.item()

        try:
            return super().default(o)
        except:
            return o


class Print(Subscriber):
    def __init__(self, model_str):
        self.model = model_str
        super().__init__()

    def handle(self, results):
        results = Subscriber._apply_translation(
            self.modelspec.get_translations(),
            results
        )
        results["MODEL"]["subset_info"] = self.subset_info
        results["MODEL"]["formula"] = self.model

        print(json.dumps(results, cls=JSONEncoder))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        help=("A formula of the form 'y ~ x1 + x2' or a file containing one "
              "formula per line.")
    )

    parser.add_argument(
        "--cohort",
        help="The cohort.",
        required=True
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

    parser.add_argument(
        "--log-errors",
        help="Log statistical test errors instead of raising.",
        action="store_true"
    )

    main(parser.parse_args())
