
import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# token characteristics
VOWELS = set("aeiou")
ID_LIKE_TOKENS = {"id", "pid", "uid", "cid", "eid", "gid"}
CAMEL_REGEX = re.compile( # camel case identifier
    r"""
    [A-Z]+(?=[A-Z][a-z]|[0-9]|\b) |  # groups of capitals before CamelCase or digit
    [A-Z]?[a-z]+ |                   # words beginning with optional capital
    [0-9]+ |                         # numbers
    [A-Z]+                           # remaining capitals
    """,
    re.VERBOSE,
)

# data structures
@dataclass
class TokenFeatures:
    token: str
    original: str
    length: int
    num_vowels: int
    num_consonants: int
    is_abbreviation: bool
    is_numeric: bool
    looks_like_id: bool
    is_all_caps: bool
    is_single_letter: bool
    has_mixed_digits: bool
@dataclass
class NameFeatures:
    original_name: str
    normalized_name: str
    tokens: List[TokenFeatures]
    num_tokens: int
    has_digit_token: bool
    has_abbreviation_token: bool
    has_vowel_rich_token: bool
    case_pattern: str
    alpha_ratio: float


# generate token features
def compute_token_features(original_token: str) -> TokenFeatures:
    norm = original_token.lower()
    is_numeric = norm.isdigit()
    length = len(norm)

    num_vowels = sum(1 for c in norm if c in VOWELS)
    num_consonants = sum(1 for c in norm if c.isalpha() and c not in VOWELS)

    is_abbrev = (
        not is_numeric
        and length <= 4
        and (num_consonants >= num_vowels)
    )

    looks_like_id = norm in ID_LIKE_TOKENS
    is_all_caps = original_token.isupper()
    is_single_letter = len(original_token) == 1 and original_token.isalpha()
    has_mixed_digits = any(c.isdigit() for c in norm) and any(c.isalpha() for c in norm)

    return TokenFeatures(
        token=norm,
        original=original_token,
        length=length,
        num_vowels=num_vowels,
        num_consonants=num_consonants,
        is_abbreviation=is_abbrev,
        is_numeric=is_numeric,
        looks_like_id=looks_like_id,
        is_all_caps=is_all_caps,
        is_single_letter=is_single_letter,
        has_mixed_digits=has_mixed_digits,
    )

# generate name features
def analyze_name(name: str) -> NameFeatures:
    normalized = name.lower()
    raw_tokens = split_camel_and_underscores(name)
    token_feats = [compute_token_features(tok) for tok in raw_tokens]

    has_digit_token = any(t.is_numeric for t in token_feats)
    has_abbrev = any(t.is_abbreviation for t in token_feats)
    has_vowel_rich = any(t.num_vowels >= 2 for t in token_feats)

    case_pattern = infer_case_pattern(name)

    # compute alpha ratio (ignore underscores)
    chars = [c for c in name if c != "_"]
    num_alpha = sum(c.isalpha() for c in chars)
    alpha_ratio = num_alpha / len(chars) if chars else 0.0

    return NameFeatures(
        original_name=name,
        normalized_name=normalized,
        tokens=token_feats,
        num_tokens=len(token_feats),
        has_digit_token=has_digit_token,
        has_abbreviation_token=has_abbrev,
        has_vowel_rich_token=has_vowel_rich,
        case_pattern=case_pattern,
        alpha_ratio=alpha_ratio,
    )

# which operations are feasible given name features
def feasible_ops(nf: NameFeatures) -> Dict[str, bool]:
    """
    Determine feasibility of each ambiguity operator for a given name,
    using enhanced token + name features.
    """

    # -----------------------------
    # Global exclusion conditions
    # -----------------------------
    # If almost no alphabetic content → no ABBREV or VOWEL_DROP
    low_alpha = nf.alpha_ratio < 0.5

    # -----------------------------
    # OP_ABBREV feasibility
    # -----------------------------
    abbrev_feasible = False

    if not low_alpha:
        for t in nf.tokens: # must be a normal alphabetic token
            if t.is_numeric or t.has_mixed_digits: 
                continue
            if t.is_single_letter:
                continue
            if t.is_all_caps:
                continue
            if t.is_abbreviation:
                continue
            if t.looks_like_id:
                continue

            if t.length >= 4: # needs length of 4
                abbrev_feasible = True
                break

    # -----------------------------
    # OP_VOWEL_DROP feasibility
    # -----------------------------
    vowel_drop_feasible = False

    if not low_alpha:
        for t in nf.tokens:
            if t.is_numeric or t.has_mixed_digits:
                continue
            if t.is_single_letter:
                continue
            if t.is_all_caps:
                continue
            if t.looks_like_id:
                continue
            
            if (t.length >= 3) and (t.num_vowels >= 2): # needs enough vowels and length
                vowel_drop_feasible = True
                break

    # -----------------------------
    # OP_CASE_FLATTEN feasibility
    # -----------------------------
    case_flatten_feasible = ( # multi-token or case-structured names
        nf.num_tokens > 1
        or nf.case_pattern in {"camel", "snake", "mixed"}
    )

    # -----------------------------
    # OP_NOISE_WRAP feasibility
    # -----------------------------
    noise_wrap_feasible = True # always feasible

    return {
        "OP_ABBREV": abbrev_feasible,
        "OP_VOWEL_DROP": vowel_drop_feasible,
        "OP_CASE_FLATTEN": case_flatten_feasible,
        "OP_NOISE_WRAP": noise_wrap_feasible,
    }



# utils

def split_camel_and_underscores(name: str) -> List[str]:
    """Split on underscores and camelCase / PascalCase boundaries."""
    parts = re.split(r'[_\s]+', name) # split on underline and whitespace
    tokens: List[str] = []

    for part in parts:
        if not part:
            continue
        for m in CAMEL_REGEX.finditer(part):
            tokens.append(m.group(0))
    return tokens

def infer_case_pattern(name: str) -> str:
    """Get the current pattern of name."""
    if "_" in name:
        return "snake"
    if name.isupper():
        return "upper"
    if name.islower():
        return "lower"
    if re.search(r"[a-z][A-Z]", name):
        return "camel"
    return "mixed"



