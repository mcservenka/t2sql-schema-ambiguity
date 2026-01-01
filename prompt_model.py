import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from models.prompt import Prompter
from models.schema_builder import SchemaBuilder
from configs.paths import SPIDER_DEV_PATH, BIRD_DEV_PATH, KAGGLEDBQA_DEV_PATH, RESULTS_PATH

load_dotenv()

MODELS = {
    "gpt-5.2": {"provider": "openai", "model": "gpt-5.2"},
    "llama-3.3-70B": {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["spider", "bird"], default="spider")
    parser.add_argument("--level", type=str, choices=["L0", "L1", "L2", "L3"], default="L0")
    parser.add_argument("--model", type=str, choices=["gpt-5.2", "llama-3.3-70B"], default="gpt-5.2")
    args = parser.parse_args()


    DATASET = args.dataset
    LEVEL = args.level
    MODEL = args.model

    # target dir
    if LEVEL == "L0":
        if DATASET == "spider": dev_path = SPIDER_DEV_PATH
        elif DATASET == "bird": dev_path = BIRD_DEV_PATH
        elif DATASET == "kaggledbqa": dev_path = KAGGLEDBQA_DEV_PATH
        else: raise Exception("Invalid dataset selection.")
    else:
        dev_path = f"data/datasets/{DATASET}_{LEVEL}/dev.json"

    # load questions
    with open(dev_path, "r") as f: 
        samples = json.load(f)

    schema_strings = {}

    responses = []

    # json as main results file
    json_path = f"{RESULTS_PATH}{DATASET}_{LEVEL}_{MODEL}_results.json"
    if os.path.exists(json_path):
        raise Exception("Responses already generated.")

    # jsonl as backup
    jsonl_path = f"{RESULTS_PATH}{DATASET}_{LEVEL}_{MODEL}_results.jsonl"
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                responses.append(json.loads(line))

    # with open(jsonl_path, "w", encoding="utf-8"): pass # create new empty jsonl backup file
    jsonl_out = open(jsonl_path, "a", encoding="utf-8")

    if len(responses) == len(samples):
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=4)
        raise Exception("Responses already generated.")

    start_index = len(responses)
    print(f"Starting generating responses at index {start_index}")

    for i, sample in tqdm(enumerate(samples[start_index:], start=start_index)):

        db_id = sample["db_id"]

        if db_id not in schema_strings:
            sb = SchemaBuilder(dataset=DATASET, db_id=db_id, level=LEVEL)
            sb.load_schema_json(repopulate_attributes=True)
            schema_strings[db_id] = sb.generate_schema_string()
        
        p = Prompter(
            provider=MODELS[MODEL]["provider"], model=MODELS[MODEL]["model"], schema_string=schema_strings[db_id]
        )

        # print(f"Generating response {i}")
        response = p.ask_question(question=sample["question"]) # returns llm response dictionary

        # determine gold sql attribute
        if DATASET == "bird" and LEVEL == "L0":
            sql_gold_attr = "SQL"
        else:
            sql_gold_attr = "query"

        response["sql_gold"] = sample[sql_gold_attr]
        response["db_id"] = db_id
        response["index"] = i

        responses.append(response)

        jsonl_out.write(json.dumps(response) + "\n")
        jsonl_out.flush()

    jsonl_out.close()

    # create final json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)

    print(f"✅ Results of {DATASET} in level {LEVEL} saved to {json_path}")


