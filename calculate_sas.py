import os
import json
import fasttext
from statistics import mean

from models.faiss_index import EmbeddingIndex
from models.sas import SchemaAmbiguityScorer
from utils.vocab import build_dictionary_from_fasttext
from configs.paths import SCHEMAS_PATH
from utils.naming import split_camel_and_underscores


# load model
model = fasttext.load_model("cc.en.300.bin")

# build dictionary and FAISS index
dict_words, dict_vectors = build_dictionary_from_fasttext(model)
index = EmbeddingIndex(dict_words, dict_vectors)

# use tokenizer function
def my_tokenizer(name: str):
    tokens = split_camel_and_underscores(name)
    tokens = [token.lower() for token in tokens] # fastText embeddings are lowercase by default
    return tokens

# build scorer
scorer = SchemaAmbiguityScorer(model, index, my_tokenizer)

# set dataset variants
levels = ["L0", "L1", "L2", "L3"]
datasets = ["spider", "bird", "kaggledbqa"]

results = {}

# load schema JSON and compute SAS for each variant
for dataset in datasets:
    results[dataset] = {}
    for level in levels:
        results[dataset][level] = {}

        # target dir
        if level == "L0":
            s_path = f"{SCHEMAS_PATH}{dataset}/"
        else:
            s_path = f"{SCHEMAS_PATH}{dataset}_{level}/"

        for db_json in os.listdir(s_path):
            with open(f"{s_path}{db_json}", "r") as f: 
                schema_json = json.load(f)

            # compute SAS
            result = scorer.schema_sas(schema_json)
            results[dataset][level][db_json] = result


# beautify results dict
final = {} 
for dataset, levels in results.items():
    for level, dbs in levels.items():
        schemas = [
            db_info["SAS_schema"]
            for db_info in dbs.values() # each individual database
        ]
        avg_schema = mean(schemas)
        final[(dataset, level)] = avg_schema


# print final results
for (dataset, level), avg in final.items():
    print(f"{dataset} {level}: {avg:.4f}")

