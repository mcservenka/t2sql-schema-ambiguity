import hashlib


def deterministic_float(*values) -> float:
    # deterministically map a tuple of values to a float in [0, 1)
    # operator selection is reproducible across runs for the same (dataset, db_id, name)

    text = "::".join(str(v) for v in values)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()    
    return int(h[:8], 16) / 2**32 # take first 8 hex chars -> 32 bits -> int -> [0,1)