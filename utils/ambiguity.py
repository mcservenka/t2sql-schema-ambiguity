import json
import numpy as np
from typing import List, Callable

EPSILON = 0.01 # lower bound for product of token scores to fix the 0 problem
ANCHOR_PATH = "configs/token_ambiguity_anchors.json"
with open(ANCHOR_PATH, "r") as f:
    anchors = json.load(f)

A_CLEAR = anchors["A_clear_mean"] # ~ 0.0
A_NOISE = anchors["A_noise_mean"] # ~ 0.30

def token_ambiguity_raw(token: str, embedding_model, faiss_index) -> float:
    """
    Compute raw ambiguity A_raw(t) = (1 - sim_max) / 2,
    without scaling or epsilon.
    """
    token = token.strip()

    # purely numeric tokens carry no meaning
    if token.isnumeric():
        return 1.0

    # get embedding
    vec = embedding_model.get_word_vector(token)

    # nearest dictionary word
    nearest_word, sim = faiss_index.nearest_word(vec)

    # convert similarity in [-1,1] to ambiguity [0,1]
    A_raw = min((1.0 - sim), 1.0)
    return float(max(0.0, A_raw))


def token_ambiguity(token: str, embedding_model, faiss_index) -> float:
    """
    Normalized token ambiguity:

        A_scaled = (A_raw - A_clear) / (A_noise - A_clear)
        A_scaled ∈ [0,1]
        A_final  = max(A_scaled, EPSILON)

    """
    A_raw = token_ambiguity_raw(token, embedding_model, faiss_index)

    # avoid division-by-zero
    if A_NOISE == A_CLEAR:
        A_scaled = A_raw
    else:
        # linear scaling
        A_scaled = (A_raw - A_CLEAR) / (A_NOISE - A_CLEAR)

    # to [0,1]
    A_scaled = float(max(0.0, min(1.0, A_scaled)))
    
    # epsilon floor
    A_final = max(A_scaled, EPSILON)
    return A_final


def name_sas(
    name: str,
    tokenizer: Callable[[str], List[str]],
    embedding_model,
    faiss_index
) -> float:
    """
    Compute SAS(name) = product of token-level ambiguities.
    """
    tokens = tokenizer(name)
    if not tokens:
        return 1.0  # empty name = max ambiguity (rare case)

    A_vals = [token_ambiguity(t, embedding_model, faiss_index) for t in tokens]

    # Product of ambiguities
    sas_value = float(np.prod(A_vals))

    # Safety bound
    return sas_value
