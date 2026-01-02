import numpy as np

# builds a dictionary V from the fastText vocabulary
def build_dictionary_from_fasttext(
    ft_model,
    max_words: int = 200000, # limit vocabulary
    min_len: int = 3, # ignore too-short tokens
    alpha_only: bool = True # keep only alphabetic words
):
    # returns:
    # - dict_words: list of strings
    # - dict_vectors: np.ndarray of shape (N, 300)

    vocab = ft_model.get_words()

    # optionally limit the number of words
    if max_words is not None:
        vocab = vocab[:max_words]

    dict_words = []
    dict_vectors = []

    for word in vocab:
        if alpha_only and not word.isalpha():
            continue
        if len(word) < min_len:
            continue

        vector = ft_model.get_word_vector(word)
        dict_words.append(word)
        dict_vectors.append(vector)

    dict_vectors = np.array(dict_vectors, dtype=np.float32)

    print(f"Dictionary built: {len(dict_words)} words, shape = {dict_vectors.shape}")
    return dict_words, dict_vectors
