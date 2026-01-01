import json
import numpy as np
import random
import fasttext

from models.faiss_index import EmbeddingIndex
from utils.ambiguity import token_ambiguity
from utils.vocab import build_dictionary_from_fasttext


SEED = 42
N_CLEAR = 5000 # number of real English tokens to sample
N_NOISE = 5000 # number of random noise tokens
MODEL_PATH = "cc.en.300.bin"
OUTPUT_PATH = "configs/token_ambiguity_anchors.json"

# generate random noise token
def generate_random_token(min_len=5, max_len=12):
    letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    L = random.randint(min_len, max_len)
    return "".join(random.choice(letters) for _ in range(L))


def main():

    # seeding for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # load fastText model
    print("Loading fastText model...")
    model = fasttext.load_model(MODEL_PATH)

    # build dictionary
    dict_words, dict_vectors = build_dictionary_from_fasttext(
        model,
        max_words=200000,
        min_len=3,
        alpha_only=True
    )

    # build FAISS index
    index = EmbeddingIndex(dict_words, dict_vectors)

    # sample clear and valid English words
    clear_words = random.sample(dict_words, N_CLEAR)

    A_clear = []
    for w in clear_words:
        A = token_ambiguity(w, model, index)
        A_clear.append(A)

    A_clear_mean = float(np.mean(A_clear))
    A_clear_std = float(np.std(A_clear))

    print(f"Mean A(clear) = {A_clear_mean:.6f} (std = {A_clear_std:.6f})")

    
    
    # generate noise tokens
    print(f"Generating {N_NOISE} random noise tokens...")
    noise_words = [generate_random_token() for _ in range(N_NOISE)]

    A_noise = []
    for w in noise_words:
        A = token_ambiguity(w, model, index)
        A_noise.append(A)

    A_noise_mean = float(np.mean(A_noise))
    A_noise_std = float(np.std(A_noise))

    print(f"Mean A(noise) = {A_noise_mean:.6f} (std = {A_noise_std:.6f})")

    # store results
    anchors = {
        "seed": SEED,
        "N_CLEAR": N_CLEAR,
        "N_NOISE": N_NOISE,
        "A_clear_mean": A_clear_mean,
        "A_clear_std": A_clear_std,
        "A_noise_mean": A_noise_mean,
        "A_noise_std": A_noise_std,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(anchors, f, indent=4)

    print(f"\nCalibration complete. Anchors saved to:")
    print(f"{OUTPUT_PATH}")


if __name__ == "__main__":
    main()