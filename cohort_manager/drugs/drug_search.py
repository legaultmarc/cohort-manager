"""
Utilities to parse free text and find drug names.
"""

import multiprocessing
import functools

from .c_drug_search import align_score
from .chembl import ChEMBL


DRUG_DB = None
POOL = None
DEFAULT_MIN_SCORE = 0.85


def find_drugs_in_query(query, min_score=DEFAULT_MIN_SCORE):
    """Searches the query to find any element of drugs."""
    if not DRUG_DB:
        _init_drug_db()
    if not POOL:
        _init_pool()

    query = query.upper()
    manager = multiprocessing.Manager()
    out = manager.list()
    f = functools.partial(_multiprocessing_search, query, out, min_score)

    POOL.map(f, DRUG_DB)

    return list(out)


def _multiprocessing_search(query, out, min_score, tu):
    molregno, name = tu
    score = align_score(query, name)
    if score >= min_score:
        out.append((molregno, name, score))


def find_drugs_in_queries(queries, min_score=DEFAULT_MIN_SCORE):
    """Cached version of find_drugs_in_query."""
    cache = {}

    out = []
    for query in queries:
        if query in cache:
            out.append(cache[query])
        result = find_drugs_in_query(query, min_score)
        out.append(result)
        cache[query] = result

    return out
    

def _init_drug_db():
    global DRUG_DB
    with ChEMBL() as db:
        DRUG_DB = db.get_all_drugs()


def _init_pool():
    global POOL
    POOL = multiprocessing.Pool(multiprocessing.cpu_count())
