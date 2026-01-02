
import os
import json
from tqdm import tqdm
from collections import Counter
from func_timeout import func_timeout

from configs.paths import RESULTS_PATH, SPIDER_DATABASE_PATH, BIRD_DATABASE_PATH, KAGGLEDBQA_DATABASE_PATH
from external.testsuitesqleval.exec_eval import eval_exec_match, eval_exec_match_with_error
from external.bird.evaluation import execute_sql, soft_execution_acc

ERROR_CATEGORIES = {
    "SCHEMA_TABLE_ERROR": "EXECUTION_ERROR",
    "SCHEMA_COLUMN_ERROR": "EXECUTION_ERROR",
    "SCHEMA_AMBIGUITY_ERROR": "EXECUTION_ERROR",
    "SYNTAX_ERROR": "EXECUTION_ERROR",
    "DIALECT_ERROR": "EXECUTION_ERROR",
    "OTHER_RUNTIME_ERROR": "EXECUTION_ERROR",
    "UNKNOWN_EXECUTION_ERROR": "EXECUTION_ERROR",
    "LLM_SYSTEM_ERROR": "INVALID_OUTPUT",
    "UNKNOWN": "INVALID_OUTPUT",
    "RESULT_MATCH_ERROR": "INCORRECT_RESULT",
}
   

class Evaluator:

    def __init__(self, dataset:str=None, level:str=None, model:str=None):

        self.dataset = dataset
        self.level = level
        self.model = model

        if level == "L0":
            if dataset == "spider": self.db_path = SPIDER_DATABASE_PATH
            elif dataset == "bird": self.db_path = BIRD_DATABASE_PATH
            elif dataset == "kaggledbqa": self.db_path = KAGGLEDBQA_DATABASE_PATH
            else: raise Exception("Invalid dataset selection.")
        else:
            self.db_path = f"data/datasets/{dataset}_{level}/database/"

        with open(f"{RESULTS_PATH}{self.dataset}_{self.level}_{self.model}_results.json", "r") as f: 
            self.results = json.load(f)
        
        self.eval_path = f"{RESULTS_PATH}{self.dataset}_{self.level}_{self.model}_eval.json"


    def score_sql(self):
        
        if os.path.exists(self.eval_path):
            raise Exception("Evaluation files already generated")

        total_score = 0

        for _, result in tqdm(enumerate(self.results)):

            db_id = result.get("db_id")
            gold_sql = result.get("sql_gold")
            pred_sql = result.get("response", {}).get("sql")
            exec_score, soft_exec_score, error_code = self.execution_accuracy(db_id=db_id, gold_sql=gold_sql, pred_sql=pred_sql)

            total_score += exec_score

            result["execution_accuracy"] = exec_score
            result["soft_execution_accuracy"] = soft_exec_score
            result["error_code"] = error_code
        
        with open(self.eval_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

        print(f"EXA for {self.model} in {self.dataset}_{self.level}: {total_score / len(self.results)}")
        return total_score / len(self.results)


    def execution_accuracy(self, db_id:str, gold_sql:str, pred_sql:str):
        
        if pred_sql is None or pred_sql == "":
            return 0, 0, "LLM_SYSTEM_ERROR"

        db = f"{self.db_path}{db_id}/{db_id}.sqlite"

        if self.dataset == "spider" or self.dataset == "kaggledbqa":
            try:
                exec_score, error_code = func_timeout(30, eval_exec_match_with_error, args=(db, pred_sql, gold_sql, False, True, False))
            except:
                exec_score = 0
                error_code = "UNKNOWN"
        elif self.dataset == "bird":            
            try:
                exec_score, error_code = func_timeout(30, execute_sql, args=(pred_sql, gold_sql, db))
            except:
                exec_score = 0
                error_code = "UNKNOWN"
        else:
            raise Exception("Uknown dataset during evaluation.")
        
        # regardless of dataset
        try:
            soft_exec_score = func_timeout(30, soft_execution_acc, args=(pred_sql, gold_sql, db))
        except:
            soft_exec_score = 0
        
        return exec_score, soft_exec_score, error_code


    # exa
    def analyze_exa(self):
        # eval file needs to be created first
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f: 
                eval = json.load(f)
        else:
            raise Exception(f"Create eval file for {self.dataset} {self.level} with fit_sql() first.")
        
        exa = 0
        soft_exa = 0
        exa_count = 0
        for sample in eval:
            exa += sample["execution_accuracy"]
            soft_exa += sample["soft_execution_accuracy"]
            exa_count += 1
        
        total_exa = round(exa / exa_count * 100, 2)
        total_soft_exa = round(soft_exa / exa_count * 100, 2)
        print(f"{self.dataset} | {self.level} | {self.model} | ExA: {total_exa} | Soft-ExA: {total_soft_exa}")
    
    def analyze_errors(self, layer1=False):
        # eval file needs to be created first
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f: 
                eval = json.load(f)
        else:
            raise Exception(f"Create eval file for {self.dataset} {self.level} with fit_sql() first.")
        
        failed = [item for item in eval if item.get("execution_accuracy") == 0]

        error_code_counts = Counter(item.get("error_code") for item in failed)
        error_category_counts = {}

        error_ratio = round(len(failed) / len(eval) * 100, 2)

        if layer1:
            #print(f"{self.dataset} | {self.level} | {self.model} | Count: {len(failed)} of {len(eval)}")
            for error_code, count in error_code_counts.items():
                category = ERROR_CATEGORIES[error_code]
                error_category_counts[category] = error_category_counts.get(category, 0) + count
                # print(f"{category}({error_code}): {count}")
            #print("-----------------------------------------")

            return error_category_counts
        
        #print(f"{self.dataset} | {self.level} | {self.model} | Error Ratio: {error_ratio}")
        for error_code, count in error_code_counts.items():
            error_type_ratio = round(count / len(failed) * 100, 2)
            #print(f"  {error_code}: {count} | {error_type_ratio}%")
        #print("-----------------------------------------")

        return error_code_counts