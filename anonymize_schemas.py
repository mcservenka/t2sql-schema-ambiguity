import os
import json
import argparse
from tqdm import tqdm

from utils.sql import verify_sample
from models.schema_anonymizer import SchemaAnonymizer
from configs.paths import SPIDER_DATABASE_PATH, BIRD_DATABASE_PATH, KAGGLEDBQA_DATABASE_PATH

"""

    creates anonymized databases for each db_id
    and stores them as new datasets in data/datasets/

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["spider", "bird", "kaggledbqa"], default="spider")
    parser.add_argument("--level", type=str, choices=["L1", "L2", "L3"], default="L1")
    args = parser.parse_args()

    DATASET = args.dataset
    ANON_LEVEL = args.level

    # set database path
    if DATASET == "spider":
        databases = os.listdir(SPIDER_DATABASE_PATH)
    elif DATASET == "bird":
        databases = os.listdir(BIRD_DATABASE_PATH)
    elif DATASET == "kaggledbqa":
        databases = os.listdir(KAGGLEDBQA_DATABASE_PATH)
    else:
        raise ValueError("Unknown Dataset selected.")


    print(f"Starting schema generation for {DATASET}.")
    new_samples = []
    for db in tqdm(databases):
        anon = SchemaAnonymizer(dataset=DATASET, db_id=db)
        mapping = anon.generate_mapping(level=ANON_LEVEL)
        anon.save_mapping()
        anon.recreate_database()
        new_samples.extend(anon.recreate_samples())

    print("Verifying new samples.")
    for sample in tqdm(new_samples):
        db_id = sample["db_id"]
        db_path = f"data/datasets/{DATASET}_{ANON_LEVEL}/database/{db_id}/{db_id}.sqlite"
        res = verify_sample(sql=sample["query"], db_path=db_path)

        if not res:
            print(f"Error executing: {db_id} -- {sample['query']}")


    with open(anon.dev_path_new, "w", encoding="utf-8") as f:
        json.dump(new_samples, f, indent=4)


