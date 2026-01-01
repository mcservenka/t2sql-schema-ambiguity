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
python build_schemas.py \
    --dataset "spider"
```

### Schema Anonymization Procedure
Next we will create the different variants of the original datasets with varying levels of ambiguity. Optionally you can run `token_level_scaling.py` beforehand. This will produce the file `/configs/token_ambiguity_anchors.json`, which contains the anchor values (`anchor_clear ~ 0.0`, `anchor_noise ~ 0.6`) that are used to linearly scale the ambiguity score (see section 4.2 in the paper). Nevertheless, this step is optional as that was already created.  

In order to generate a modified dataset variant you can run:
```
python anonymize_schemas.py \
    --dataset "spider" \
    --level "L1"
```
Select the specific dataset (`spider`, `bird`, `kaggledbqa`) as well as the level of obfuscation (`L1`, `L2`, `L3`). This creates a mapping connecting each original name to the anonymized string (in `/data/mappings/`) and generates the entire modified dataset (in `/data/datasets/`) including sqlite-databases and the gold queries of the particular development set.  
Since we want to use the newly constructed database schema for further processing we need to run `build_schemas.py` again. This time, set the `level` parameter accordingly:
```
python build_schemas.py \
    --dataset "spider" \
    --level "L1"
```

### Schema Ambiguity Score (SAS)
Once you have created all dataset versions you can calculate their specific Schema Ambiguity Scores. The SAS is designed to capture how easily schema object names can be grounded in natural language, independently of any particular model or task performance.  
Just run:
```
python calculate_sas.py
```
This will output the SAS for each dataset variant including the original. We report these scores in out paper:
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>L0</th>
      <th>L1</th>
      <th>L2</th>
      <th>L3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Spider</td>
      <td>.031</td>
      <td>.071</td>
      <td>.122</td>
      <td>.217</td>
    </tr>
    <tr>
      <td>BIRD</td>
      <td>.112</td>
      <td>.208</td>
      <td>.193</td>
      <td>.256</td>
    </tr>
    <tr>
      <td>KaggleDBQA</td>
      <td>.171</td>
      <td>.244</td>
      <td>.211</td>
      <td>.297</td>
    </tr>
  </tbody>
</table>

### Prompt Model
Now using our augmented versions of Spider, BIRD-SQL and KaggleDBQA, we can start prompting the models. In terms of LLMs utilized within this study, one open- (`gpt-5.2` via OpenAI API) and one closed-source (`llama-3.3-70B` via TogetherAI API) LLM were tested, which is common in this field of research. To generate the results in `data/results/` run the following command for each variant:
```
python prompt_model.py \
    --dataset "spider" \
    --level "L0" \
    --model "gpt-5.2" \
```
Make sure that the dataset variant you select really exists in `data/datasets/` and `data/schemas/`, respectively.

### Evaluation
Eventually, you can evaluate the responses by running `evaluate_results.py`. This will add the evaluation scores to your response objects and create a new file in `data/results/` and print the evaluation results to the console.
```
python evaluate_results.py \
    --dataset "spider" \
    --level "L0" \
    --model "gpt-5.2" \
```
Again make sure the results for the selected variants were generated beforehand.

## Experiment Results
Down below we illustrated the official results of our paper. Please note that - although our schema schema anonymizer is inherently deterministic - the results may vary after rerunning the experiment due to the inherent stochasticity of the LLM. For detailed evaluation results feel free to check out section 7 of the paper.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Level</th>
      <th>GPT-5.2</th>
      <th>Llama-3.3-70B</th>
    </tr>
  </thead>
  <tbody>
    <!-- Spider -->
    <tr>
      <td><strong>Spider</strong></td>
      <td>L0</td>
      <td>76.79</td>
      <td>76.21</td>
    </tr>
    <tr>
      <td><strong>Spider</strong></td>
      <td>L1</td>
      <td>76.02</td>
      <td>74.37</td>
    </tr>
    <tr>
      <td><strong>Spider</strong></td>
      <td>L2</td>
      <td>75.44</td>
      <td>73.98</td>
    </tr>
    <tr>
      <td><strong>Spider</strong></td>
      <td>L3</td>
      <td>76.40</td>
      <td>71.76</td>
    </tr>
    <!-- BIRD -->
    <tr>
      <td><strong>BIRD</strong></td>
      <td>L0</td>
      <td>36.57</td>
      <td>35.72</td>
    </tr>
    <tr>
      <td><strong>BIRD</strong></td>
      <td>L1</td>
      <td>35.01</td>
      <td>33.96</td>
    </tr>
    <tr>
      <td><strong>BIRD</strong></td>
      <td>L2</td>
      <td>34.68</td>
      <td>34.49</td>
    </tr>
    <tr>
      <td><strong>BIRD</strong></td>
      <td>L3</td>
      <td>33.38</td>
      <td>33.05</td>
    </tr>
    <!-- KaggleDBQA -->
    <tr>
      <td><strong>KaggleDBQA</strong></td>
      <td>L0</td>
      <td>19.69</td>
      <td>43.98</td>
    </tr>
    <tr>
      <td><strong>KaggleDBQA</strong></td>
      <td>L1</td>
      <td>19.91</td>
      <td>42.23</td>
    </tr>
    <tr>
      <td><strong>KaggleDBQA</strong></td>
      <td>L2</td>
      <td>19.26</td>
      <td>36.54</td>
    </tr>
    <tr>
      <td><strong>KaggleDBQA</strong></td>
      <td>L3</td>
      <td>19.04</td>
      <td>43.11</td>
    </tr>
  </tbody>
</table>

## Citation
```citation
@article{schema-size-matters,
    author  =   {Cservenka Markus},
    title   =   {Schema Ambiguity in Text-to-SQL: Measurement, Benchmarks, and Robustness},
    year    =   {2025}
}
