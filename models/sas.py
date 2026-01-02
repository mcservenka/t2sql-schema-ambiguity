# utils/schema_sas.py

from typing import Dict, Any, Callable, List
from utils.ambiguity import name_sas


class SchemaAmbiguityScorer:
    """
    Computes SAS scores for database objects and an entire schema.
    Uses calibrated token ambiguity under the hood.
    """

    def __init__(self, embedding_model, faiss_index, tokenizer: Callable[[str], List[str]]):
        # embedding_model: fastText model (loaded)
        # faiss_index: FAISS index over dictionary embeddings
        # tokenizer: function that converts name -> list of tokens
        
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.tokenizer = tokenizer

    def db_object_sas(self, object_name: str) -> float:
        return name_sas(
            object_name,
            self.tokenizer,
            self.embedding_model,
            self.faiss_index
        )

    def schema_sas(self, schema_json: Dict[str, Any]) -> Dict[str, float]:
        schema_dict = schema_json["schema"]

        table_scores = []
        column_scores = []

        for table_name, table_data in schema_dict.items():
            # 1. Table-level SAS
            t_score = self.db_object_sas(table_name)
            table_scores.append(t_score)

            # 2. Column-level SAS (with context)
            for col in table_data["columns"]:
                col_name = col["name"]
                c_score = self.db_object_sas(col_name)
                column_scores.append(c_score)

        # Compute aggregated SAS
        SAS_tables = float(sum(table_scores) / len(table_scores))
        SAS_columns = float(sum(column_scores) / len(column_scores))
        SAS_schema = float((sum(table_scores) + sum(column_scores)) / (len(table_scores) + len(column_scores)))

        return {
            "SAS_tables": SAS_tables,
            "SAS_columns": SAS_columns,
            "SAS_schema": SAS_schema
        }
