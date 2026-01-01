# Schema Ambiguity in Text-to-SQL: Measurement, Benchmarks, and Robustness

Although strong performance has been achieved on Text-to-SQL established benchmarks, these benchmarks largely assume linguistically explicit and human-readable database schemas. In real-world deployments, however, schema object names are often abbreviated or compressed, which leads to linguistic ambiguity and weakens lexical alignment between natural language questions and database structures. This repository provides the source code for the paper **Schema Ambiguity in Text-to-SQL: Measurement, Benchmarks, and Robustness**, which addresses this gap by introducing a deterministic schema transformation pipeline as well as the Schema Ambiguity Score (SAS), an embedding-based metric that measures the linguistic explicitness of schema object names. On the basis of these contributions multi-level variants of Spider, BIRD-SQL, and KaggleDBQA are evaluated using leading LLMs. In order to reproduce the results, follow the instructions below.

>Cservenka, Markus. "Schema Ambiguity in Text-to-SQL: Measurement, Benchmarks, and Robustness", 2025.

Link to paper following soon...

## Ressources
To set up the environment, start by downloading the development sets of [Spider](https://yale-lily.github.io/spider), [BIRD-SQL](https://bird-bench.github.io/) and [KaggleDBQA](https://github.com/Chia-Hsuan-Lee/KaggleDBQA) to the folders `./data/datasets/spider/`, `./data/datasets/bird/dev/` and `./data/datasets/kaggledbqa/`, respectively. Then add the git submodule [Test-Suite-Evaluation](https://github.com/taoyds/test-suite-sql-eval) to  `./external/` and copy the BIRD's evaluation module `[evaluation.py](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py)` into `./external/bird/`. We will need them later for computing the execution accuracy of answerable samples. Make sure to define the OpenAI and TogetherAI API keys in your environment variables as `OPENAI_API_KEY`, `OPENAI_API_ORGANIZATION`, `OPENAI_API_PROJECT` and `TOGETHERAI_API_KEY`. We also recommend using the `dotenv`-package.

## Environment Setup
Now set up the Python environment:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Next please download fastText's English Common Crawl Word Vectors `cc.en.300.bin` from [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html) and put it in the root folder.

## Experiment
Follow the steps down below to recreate the experiment.

### Schema Representation
First we need to build the schema representation objects for the original datasets' databases using `build_schemas.py`. This will store each database schema of each dataset (Spider, BIRD-SQL, KaggleDBQA) as a json-file in `data/schemas/`
```
python build_schemas.py
```

### Schema Anonymization Procedure
Next we will create the different variants of the original datasets with varying levels of ambiguity. Optionally you can run `token_level_scaling.py` beforehand. This will produce the file `/configs/token_ambiguity_anchors.json`, which contains the anchor values (`anchor_clear ~ 0.0`, `anchor_noise ~ 0.6`) that are used to linearly scale the ambiguity score (see section 4.2 in the paper). Nevertheless, this step is optional as that was already created.  

In order to generate a modified dataset variant you can run:
```
python anonymize_schemas.py \
    --dataset "spider" \
    --level "L1"
```
Select the specific dataset (`spider`, `bird`, `kaggledbqa`) as well as the level of obfuscation (`L1`, `L2`, `L3`). This generates the entire modified dataset in `/data/datasets/` including sqlite-databases and the gold queries of the particular development set.

