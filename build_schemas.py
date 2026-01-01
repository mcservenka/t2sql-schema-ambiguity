import os
import json
import argparse
from tqdm import tqdm

from models.schema_builder import SchemaBuilder
from configs.paths import SPIDER_DATABASE_PATH, BIRD_DATABASE_PATH, KAGGLEDBQA_DATABASE_PATH, SPIDER_DEV_PATH

"""

    creates schema representation in json via
    SchemaBuilder and stores files in configs.paths.SCHEMA_PATHS

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["spider", "bird", "kaggledbqa"], default="spider")
    parser.add_argument("--level", type=str, choices=["L0", "L1", "L2", "L3"], default="L0")
    args = parser.parse_args()

    DATASET = args.dataset
    ANON_LEVEL = args.level

    # set database path
    if ANON_LEVEL == "L0":
        if DATASET == "spider":
            databases = os.listdir(SPIDER_DATABASE_PATH)
        elif DATASET == "bird":
            databases = os.listdir(BIRD_DATABASE_PATH)
        elif DATASET == "kaggledbqa":
            databases = os.listdir(KAGGLEDBQA_DATABASE_PATH)
        else:
            raise ValueError("Unknown Dataset selected.")
    else:
        db_path = f"data/datasets/{DATASET}_{ANON_LEVEL}/database/"
        databases = os.listdir(db_path)

    # create schema representations for all databases (not limited to dev only)
    for db in tqdm(databases):
        with SchemaBuilder(dataset="spider", db_id=db, level=ANON_LEVEL) as sb:
            sb.build_schema_object()
            sb.save_schema_json()
