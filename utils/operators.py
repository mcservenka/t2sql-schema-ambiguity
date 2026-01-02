from typing import List

from utils.naming import NameFeatures, TokenFeatures
from utils.abbr import COMMON_ABBREVS
from utils.hashing import deterministic_float

# prefixes and suffixes for noise wrap operation
PREFIXES = {
    "L1": ["tbl_", "t_", "u_", "dim_", "d_", "f_"],
    "L2": ["vx_", "kzr_", "mpr_", "qlt_", "zhn_"],
    "L3": ["xq7_", "ztr9_", "pqm4_", "vsk8_", "kxh3_"]
}
SUFFIXES = {
    "L1": ["_tbl", "_t", "_col", "_fld", "_01", "_x"],
    "L2": ["_vx", "_kz", "_mpr", "_ql", "_zhn"],
    "L3": ["_xq7", "_z9", "_p4m", "_v8k", "_k3h"]
}

# common abbreviations for abbrev operation


def apply_operator(op: str, nf: NameFeatures, level: str) -> str:
    if op == "OP_ABBREV":
        return apply_abbrev(nf)
    elif op == "OP_VOWEL_DROP":
        return apply_vowel_drop(nf)
    elif op == "OP_CASE_FLATTEN":
        return apply_case_flatten(nf, level)
    elif op == "OP_NOISE_WRAP":
        return apply_noise_wrap(nf, level)
    elif op == "OP_IDENTITY":
        return nf.normalized_name
    else:
        raise ValueError(f"Unknown operator: {op}")



def apply_abbrev(nf: NameFeatures) -> str:
    new_tokens = []

    for t in nf.tokens:
        if t.is_numeric or t.looks_like_id or t.is_abbreviation:
            new_tokens.append(t.token)
            continue

        # common dictionary
        if t.token in COMMON_ABBREVS:
            new_tokens.append(COMMON_ABBREVS[t.token])
            continue

        # long tokens
        if t.length >= 8:
            new_tokens.append(t.token[:4])
            continue

        # mid-length tokens
        if t.length >= 4:
            new_tokens.append(t.token[:3])
            continue

        # too short -> leave unchanged
        new_tokens.append(t.token)

    # use underscores for readability
    return "_".join(new_tokens)

def apply_vowel_drop(nf: NameFeatures) -> str:
    new_tokens = []

    for t in nf.tokens:
        if t.is_numeric or t.looks_like_id or t.is_abbreviation or t.is_all_caps:
            new_tokens.append(t.token)
            continue

        chars = list(t.token)
        first = chars[0]
        rest = [c for c in chars[1:] if c not in "aeiou"]

        # enforce min length
        if len(rest) < 2:
            # try to keep one vowel
            for c in chars[1:]:
                if c in "aeiou":
                    rest.append(c)
                    break

        new_tokens.append(first + "".join(rest))

    return "_".join(new_tokens)

def apply_case_flatten(nf: NameFeatures, level: str) -> str:
    # normalized tokens already lowercase
    base = "_".join(t.token for t in nf.tokens)

    if level in ("L1", "L2"):
        return base  # snake_case form

    if level == "L3":
        return base.replace("_", "")  # full flatten

    return base

def apply_noise_wrap(nf: NameFeatures, level: str) -> str:
    base = "_".join(t.token for t in nf.tokens)

    # choose level specific
    prefs = PREFIXES[level]
    sufs = SUFFIXES[level]

    # Deterministic floats for choices
    u1 = deterministic_float(nf.original_name.lower(), level, "nw_prefix")
    u2 = deterministic_float(nf.original_name.lower(), level, "nw_suffix")
    u3 = deterministic_float(nf.original_name.lower(), level, "nw_mode")

    # Pick prefix + suffix deterministically
    prefix = prefs[int(u1 * len(prefs))]
    suffix = sufs[int(u2 * len(sufs))]

    # Decide wrapping mode: prefix only / suffix only / both
    if u3 < 0.33:
        return f"{prefix}{base}"
    elif u3 < 0.66:
        return f"{base}{suffix}"
    else:
        return f"{prefix}{base}{suffix}"
