"""
Utilities to parse free text and find drug names.
"""

import collections
import multiprocessing
import functools
import logging
import csv

from .c_drug_search import align_score
from .chembl import ChEMBL


DRUG_DB = {}
DEFAULT_MIN_SCORE = 0.85
logger = logging.getLogger(__name__)


def find_drugs_in_query(query, min_score=DEFAULT_MIN_SCORE):
    """Searches the query to find any element of drugs."""
    if not DRUG_DB:
        _init_drug_db()

    query = query.upper()

    # Search in preferred.
    out = _match_if_score(query, DRUG_DB["PREFERRED"], min_score)

    # If there are no perfect matches, we continue searching in the synonyms.
    perfect_matches = [i for i in out if i[2] == 1 and query == i[1]]

    if perfect_matches:
        return list(set(perfect_matches))

    # Look at the synonyms.
    syn_matches = _match_if_score(query, DRUG_DB["SYNONYMS"], min_score)

    perfect_matches = [i for i in syn_matches if i[2] == 1 and query == i[1]]

    if perfect_matches:
        return list(set(perfect_matches))

    out.extend(syn_matches)

    return list(set(out))


def _match_if_score(query, db, min_score):
    out = []
    for molregno, name in db:
        score = align_score(query, name)
        if score >= min_score:
            out.append((molregno, name, score))
    return out


def find_drugs_in_queries(queries, min_score=DEFAULT_MIN_SCORE):
    """Cached, multiprocessing version of find_drugs_in_query."""
    manager = multiprocessing.Manager()
    cache = manager.dict()

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    _f = functools.partial(_multiprocessing_query, cache, min_score)
    out = pool.map(_f, queries)
    pool.close()

    return list(out)


def _multiprocessing_query(cache, min_score, query):
    if query in cache:
        return cache[query]
    result = find_drugs_in_query(query, min_score)
    cache[query] = result
    return result


def remove_substring_matches(queries, results, min_length=4):
    """Remove perfect matches that are not 'words' in the query.

    As an example, remove ST (molregno 674955) from matching PRAVASTATIN.

    TODO: Fix the case where the matching drug has multiple words (e.g.
    CLOBETASOL PROPIONATE).

    """
    out = []
    for i, query in enumerate(queries):
        words = set(query.split())
        filtered_results = []
        for tu in results[i]:
            molregno, match, score = tu
            if len(match) > min_length:
                # Match is too big to eliminate (greater than min_length).
                filtered_results.append(tu)
            else:
                # Check if it is a word match.
                match = match.split()
                if not (set(words) - set(match)):
                    filtered_results.append(tu)
                else:
                    # We ignore substring matches.
                    pass

        out.append(filtered_results)

    return out


def fix_hierarchical_matches(queries, results):
    """In ChEMBL, some drugs have a hierarchical relationship. This function
    removes redundant matches.

    """
    cache = {}
    warned = set()
    out = []
    with ChEMBL() as db:
        for i, query in enumerate(queries):
            if query in cache:
                out.append(cache[query])
                continue

            filtered_results = []
            matches = collections.defaultdict(list)

            for tu in results[i]:
                matches[tu[1]].append(tu)

            for name, li in matches.items():
                if len(li) == 1:
                    # Single matching molregno.
                    filtered_results.extend(li)
                else:
                    # Multiple molregnos for the same drug name.
                    parent = set()
                    score = None
                    for tu in li:
                        score = tu[2]
                        parent.add(db.get_parent(tu[0]))
                    if len(parent) != 1:
                        if name not in warned:
                            logger.warning(
                                "Drug '{}' has multiple entries in ChEMBL "
                                "(needs curation).".format(name)
                            )
                            warned.add(name)
                    filtered_results.append((parent.pop(), name, score))
            cache[query] = filtered_results
            out.append(filtered_results)

    return out


def write_results(filename, queries, results):
    """Write the results of drug queries to disk."""
    # Write queries matches only once.
    _written = set()
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["query", "molregno", "matching_drug"])
        for i, query in enumerate(queries):
            if query in _written:
                continue

            _written.add(query)
            if results[i]:
                for tu in results[i]:
                    molregno, match, score = tu
                    row = [query, molregno, match]
                    writer.writerow(row)
            else:
                writer.writerow([query, "", ""])


def _init_drug_db():
    with ChEMBL() as db:
        DRUG_DB["PREFERRED"] = db.get_preferred_drugs()
        DRUG_DB["SYNONYMS"] = list({(i[0], i[1]) for i in db.get_synonyms()})
