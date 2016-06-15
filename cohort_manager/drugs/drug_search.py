"""
Utilities to parse free text and find drug names.
"""

from __future__ import division, print_function

import collections
import multiprocessing
import operator
import functools
import logging

import pandas as pd

from .c_drug_search import align_score as c_align_score
from .chembl import ChEMBL


DRUG_DB = {}
DEFAULT_MIN_SCORE = 0.8
logger = logging.getLogger(__name__)


def find_drugs_in_query(query, min_score=DEFAULT_MIN_SCORE):
    """Searches the query to find any element of drugs.

    TODO. It would be best to prioritize matches in the CUSTOM database to
    make it easy for users to override bad behaviour or unwanted matches.

    """
    if "PREFERRED" not in DRUG_DB or "SYNONYMS" not in DRUG_DB:
        _init_drug_db()

    query = query.upper()

    # Search in custom database.
    out = []
    if DRUG_DB.get("CUSTOM"):
        out.extend(_match_if_score(query, DRUG_DB["CUSTOM"], min_score))

    # Search in preferred.
    out.extend(_match_if_score(query, DRUG_DB["PREFERRED"], min_score))

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

    return _choose_hits(query, set(out))


def _match_if_score(query, db, min_score):
    out = []
    for molregno, name in db:
        score, left, right = c_align_score(query, name)
        normalized_score = score / (3 * len(name))
        if normalized_score >= min_score:
            out.append((molregno, name, score, left, right))
    return out


def _choose_hits(query, hits, keep_short=False):
    """Choose hits that best explain the user sequence.

    TODO. This technique can yield false positives if there are repetitions
    in the query string.

    e.g. 'dbname (dbname)' will try to use lower quality matches than 'dbname'
    to explain the second part of the string.

    """
    if not hits:
        return hits

    # Sort by the distance to the query.
    hits = sorted(hits, key=lambda x: abs(len(x[1]) - len(query)))

    # Sort again with respect to score.
    # Because sorting is guaranteed to be stable, the first store will still
    # be useful to break ties.
    hits = sorted(hits, key=operator.itemgetter(2), reverse=True)

    out = []
    explained = []
    for hit in hits:
        molregno, name, score, left, right = hit

        # Exclude very short database matches as these are mostly noise.
        if len(name) <= 3 and not keep_short:
            continue

        cur = _Segment(left, right)

        if not out:
            out.append(hit)
            explained.append(cur)
            continue

        # We add the hit if it explains something new.
        new_chars = len(name)
        for segment in explained:
            new_chars -= segment.n_overlap(cur)

        if new_chars >= 3:
            out.append(hit)
            explained.append(cur)
            explained = sorted(explained, key=lambda x: x.start)
            explained = _Segment.merge_segments(explained)

    # Hits are not satisfactory if the selection explains less than a fraction
    # of the query.
    total_explained = sum([seg.end - seg.start + 1 for seg in explained])
    if (total_explained / len(query)) < 0.4:
        return []

    return out


class _Segment(object):
    """Class that supports basic manipulation of segments.

    Adapted from gepyto (https://github.com/legaultmarc/gepyto).

    """
    def __init__(self, start, end):
        self.start = int(start)
        self.end = int(end)
        assert self.start <= self.end

    def overlaps_with(self, segment):
        return (self.start <= segment.end and
                self.end >= segment.start)

    def n_overlap(self, segment):
        if not self.overlaps_with(segment):
            return 0
        return min(self.end, segment.end) - max(self.start, segment.start) + 1

    @staticmethod
    def merge_segments(li):
        """Merge overlapping segments in a sorted list."""
        for i in range(len(li) - 1):
            cur = li[i]
            nxt = li[i + 1]
            if nxt.start < cur.start:
                raise Exception("Only sorted lists of segments can be "
                                "merged. Sort using the position first.")

        merged_segments = []
        i = 0
        while i < len(li) - 1:
            j = i

            # Walk as long as the segments are overlapping.
            cur = li[i]
            nxt = li[i + 1]
            block = [cur.start, cur.end]
            if nxt.start <= cur.end:
                block[1] = max(block[1], nxt.end)

            while nxt.start <= block[1] and j + 1 < len(li) - 1:
                block[1] = max(block[1], nxt.end)
                j += 1
                cur = li[j]
                nxt = li[j + 1]

            merged_segments.append(
                _Segment(block[0], block[1])
            )
            i = j + 1

        if li[-1].start > li[-2].end:
            merged_segments.append(li[-1])

        return merged_segments

    def __repr__(self):
        return "<_Segment object {}-{}>".format(self.start, self.end)


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


def fix_hierarchical_matches(queries, results):
    """In ChEMBL, some drugs have a hierarchical relationship. This function
    removes redundant matches.

    """
    return results
    cache = {}
    warned = set()
    out = []
    with ChEMBL() as db:
        for i, query in enumerate(queries):
            if query in cache:
                out.append(cache[query])
                continue

            filtered_results = []  # Result tuples with the fixed hierarchy.
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
    """Write the results of drug queries to disk.

    We will write an excel spreadsheet.

    """
    writer = pd.ExcelWriter(filename)

    # Count occurences for every query.
    counts = collections.Counter(queries)

    # Process results to long format.
    _added = set()
    pd_results = []
    for i, res in enumerate(results):
        if queries[i] in _added:
            continue
        _added.add(queries[i])

        if not res:
            pd_results.append([
                queries[i], "", "", "", counts.get(queries[i], -1)
            ])
        else:
            for tu in res:
                molregno, name, score = tu[:3]
                pd_results.append([
                    queries[i], molregno, name, score, counts.get(queries[i])
                ])

    df = pd.DataFrame(
        pd_results,
        columns=("query", "molregno", "name", "score", "n_occurences"),
        dtype=str
    )

    df = df.sort_values("query")

    # Write to disk.
    df.loc[
        df["molregno"] == "", ["query", "molregno", "n_occurences"]
    ].to_excel(writer, "Not Found", index=False)

    df.loc[df["molregno"] != "", :].to_excel(writer, "Curation", index=False)
    writer.save()


def add_custom_database(df):
    """Add a list of custom mapping name -> molregno.

    :param df: A dataframe with the expected columns.
    :dtype df: pandas.DataFrame

    """
    DRUG_DB["CUSTOM"] = []
    for i, row in df.iterrows():
        DRUG_DB["CUSTOM"].append((row["molregno"], row["name"]))


def _init_drug_db():
    with ChEMBL() as db:
        DRUG_DB["PREFERRED"] = db.get_preferred_drugs()
        DRUG_DB["SYNONYMS"] = list({(i[0], i[1]) for i in db.get_synonyms()})
