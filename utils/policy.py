from typing import Dict

from utils.naming import NameFeatures, feasible_ops
from utils.hashing import deterministic_float

# operator constants
OP_ABBREV = "OP_ABBREV"
OP_VOWEL_DROP = "OP_VOWEL_DROP"
OP_CASE_FLATTEN = "OP_CASE_FLATTEN"
OP_NOISE_WRAP = "OP_NOISE_WRAP"
OP_IDENTITY = "OP_IDENTITY"


# ambiguity level weights
# infeasible ops are zeroed then renormalized.
LEVEL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "L0": { # original
        OP_IDENTITY:     1.0,
        OP_ABBREV:       0.0,
        OP_VOWEL_DROP:   0.0,
        OP_CASE_FLATTEN: 0.0,
        OP_NOISE_WRAP:   0.0,
    },
    "L1": { # mild
        OP_IDENTITY:     0.55,
        OP_ABBREV:       0.10,
        OP_VOWEL_DROP:   0.00,
        OP_CASE_FLATTEN: 0.25,
        OP_NOISE_WRAP:   0.10
    },
    "L2": { # medium
        OP_IDENTITY:     0.20,
        OP_ABBREV:       0.30,
        OP_VOWEL_DROP:   0.15,
        OP_CASE_FLATTEN: 0.20,
        OP_NOISE_WRAP:   0.15
    },
    "L3": { # heavy
        OP_IDENTITY:     0.05,
        OP_ABBREV:       0.40,
        OP_VOWEL_DROP:   0.25,
        OP_CASE_FLATTEN: 0.10,
        OP_NOISE_WRAP:   0.20
    }
}


def choose_operator(level: str, nf: NameFeatures) -> str:
    """
    Given an ambiguity level and name features deterministically 
    choose one operator from the feasible set

    Returns one of:
      OP_ABBREV, OP_VOWEL_DROP, OP_CASE_FLATTEN, OP_NOISE_WRAP, OP_IDENTITY
    """

    if level not in LEVEL_WEIGHTS:
        raise ValueError(f"Unknown ambiguity level: {level}")
    
    # L0
    if level == "L0":
        return OP_IDENTITY # always identity

    base_weights = LEVEL_WEIGHTS[level]
    feas = feasible_ops(nf)

    # mask infeasible ops
    ops = [OP_IDENTITY, OP_ABBREV, OP_VOWEL_DROP, OP_CASE_FLATTEN, OP_NOISE_WRAP]
    masked_weights = []
    for op in ops:
        if op != OP_IDENTITY and not feas.get(op, False):
            masked_weights.append(0.0)
        else:
            masked_weights.append(base_weights.get(op, 0.0))

    total = sum(masked_weights)

    # if nothing feasible (or all zero weights) -> identity
    if total <= 0.0:
        return OP_IDENTITY

    # renormalize
    normalized = [w / total for w in masked_weights]

    # deterministic "random" choice via hashing
    u = deterministic_float(nf.original_name.lower(), level)

    cumulative = 0.0
    for op, w in zip(ops, normalized):
        cumulative += w
        if u <= cumulative:
            return op

    # numeric edge case (floating point) -> fallback to last feasible op
    for op, w in reversed(list(zip(ops, masked_weights))):
        if w > 0:
            return op

    return OP_IDENTITY
