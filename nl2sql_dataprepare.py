from collections import defaultdict
from typing import Dict, Tuple
import re

def group_queries_by_db(json_data, cluster_num: int = -1):
    """Group items by db_id -> {db_name: [items,...]}."""
    db_query_dict = defaultdict(list)  # {db name: [q1, q2, ...]}
    for item in json_data:
        db_name = item.get("db_id")
        if db_name is None:
            raise KeyError("Record missing 'db_id' key")
        db_query_dict[db_name].append(item)
    return db_query_dict


# Zero-width / invisible characters to strip from text
_ZW_SPACES = "".join([
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # BOM
])


def normalize_db(db: str) -> str:
    """Normalize a database name-like string."""
    if db is None:
        return ""
    db = db.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return db.strip()


def normalize_query(q: str) -> str:
    """Normalize an NL query string (remove invisibles, collapse spaces, trim quotes)."""
    if q is None:
        return ""
    # remove zero-width / invisible characters
    for z in _ZW_SPACES:
        q = q.replace(z, "")
    # unify whitespace
    q = q.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # trim outer quotes/whitespace (ASCII + smart quotes)
    q = q.strip().strip('"\'' "“”‘’")
    if q and q[0] in '"“‘':
        q = q[1:].lstrip()
    if q and q[-1] in '"”’':
        q = q[:-1].rstrip()
    # collapse multiple spaces
    q = re.sub(r"\s+", " ", q).strip()
    return q


def norm_key(db: str, q: str) -> Tuple[str, str]:
    """Return a normalized (db, query) tuple for mapping/dedup."""
    return normalize_db(db), normalize_query(q)


def db_queries_clusterID_dict(predicted_clusters, json_data):
    """
    Build a mapping {(normalized_db, normalized_query): cluster_id}.
    """
    mapping: Dict[Tuple[str, str], int] = {}
    for idx, cid in enumerate(predicted_clusters):
        db_name = json_data[idx].get("db_id")
        query = json_data[idx].get("question")
        mapping[norm_key(db_name, query)] = int(cid)
    return mapping