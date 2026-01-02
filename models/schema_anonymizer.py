import re
import os
import json
import sqlite3
import sqlglot
from sqlglot import exp
from typing import List

from utils.naming import analyze_name
from utils.policy import choose_operator
from utils.operators import apply_operator
from configs.paths import (
    SCHEMAS_PATH, MAPPINGS_PATH, SPIDER_DATABASE_PATH, 
    BIRD_DATABASE_PATH, KAGGLEDBQA_DATABASE_PATH,
    SPIDER_DEV_PATH, BIRD_DEV_PATH, KAGGLEDBQA_DEV_PATH
)

SQL_KEYWORDS = {
    "abort", "action", "add", "after", "all", "alter", "analyze", "and", "as",
    "asc", "attach", "autoincrement", "before", "begin", "between", "by",
    "cascade", "case", "cast", "check", "collate", "column", "commit",
    "conflict", "constraint", "create", "cross", "current_date",
    "current_time", "current_timestamp", "database", "default", "deferrable",
    "deferred", "delete", "desc", "detach", "distinct", "drop", "each", "else",
    "escape", "except", "exclusive", "exists", "explain", "fail", "for",
    "foreign", "from", "full", "glob", "group", "having", "if", "ignore",
    "immediate", "in", "index", "indexed", "initially", "inner", "insert",
    "instead", "intersect", "into", "is", "isnull", "join", "key", "left",
    "like", "limit", "match", "natural", "no", "not", "notnull", "null", "of",
    "offset", "on", "or", "order", "outer", "plan", "pragma", "primary",
    "query", "raise", "recursive", "references", "regexp", "reindex",
    "release", "rename", "replace", "restrict", "right", "rollback", "row",
    "savepoint", "select", "set", "table", "temp", "temporary", "then", "to",
    "transaction", "trigger", "union", "unique", "update", "using", "vacuum",
    "values", "view", "virtual", "when", "where", "with", "without"
}


class SchemaAnonymizer():


    def __init__(self, dataset:str, db_id:str):
        
        self.dataset = dataset
        self.db_id = db_id

        # original db path
        if dataset == "spider":
            self.db_path = f"{SPIDER_DATABASE_PATH}{db_id}/{db_id}.sqlite"
            self.dev_path = SPIDER_DEV_PATH
        elif dataset == "bird":
            self.db_path = f"{BIRD_DATABASE_PATH}{db_id}/{db_id}.sqlite"
            self.dev_path = BIRD_DEV_PATH
        elif dataset == "kaggledbqa":
            self.db_path = f"{KAGGLEDBQA_DATABASE_PATH}{db_id}/{db_id}.sqlite"
            self.dev_path = KAGGLEDBQA_DEV_PATH
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # new paths
        self.db_path_new = None # gets created when level is set in generate mapping
        self.dev_path_new = None # gets created when level is set in generate mapping

        # load schema representations
        with open(f"{SCHEMAS_PATH}{self.dataset}/{db_id}.json", "r") as f: 
            schema = json.load(f)
        self.schema = schema["schema"]

        # load samples
        with open(self.dev_path, "r") as f: 
            samples = json.load(f)
        self.samples = [sample for sample in samples if sample["db_id"] == self.db_id]

        self.level = None
        self.mapping = {} # maps old to new names
        self.mapping_reverse = {} # maps new to old names
        self.metadata = {} # documents metadata of mapping process
        self.dev_new = []


    #
    # mapping
    #

    # collect all names of schema
    def collect_names(self) -> List[str]:
        names = set()

        # table names
        for table in self.schema.keys():
            names.add(table)

        # column names
        for table, table_obj in self.schema.items():
            for col in table_obj["columns"]:
                names.add(col["name"])

        return sorted(names)
    
    # generate global mapping dictionary old_name to new_name
    def generate_mapping(self, level:str="L0"):
        self.level = level
        self.db_path_new = f"data/datasets/{self.dataset}_{self.level}/database/{self.db_id}/{self.db_id}.sqlite"
        self.dev_path_new = f"data/datasets/{self.dataset}_{self.level}/dev.json"
        self.mapping = {} # reset mapping
        all_names = self.collect_names()

        for name in all_names:

            canonical = name.lower()

            if canonical in self.mapping: # mapping already exists
                new_name = self.mapping[canonical]

            else:
                nf = analyze_name(name)
                op = choose_operator(level, nf)
                new_name = apply_operator(op, nf, level)

                # handle new names that appear multiple times (just add _<counter>)
                counter = 1
                while new_name in self.mapping_reverse: # check if new name already exists
                    new_name = f"{new_name}_{counter}"
                    counter += 1
                
                # handle transformed names that are valid sql keywords
                if new_name.lower() in SQL_KEYWORDS:
                    new_name = f"f_{new_name}" # just add some prefix

                self.mapping[canonical] = new_name
                self.mapping_reverse[new_name] = canonical

                self.metadata[canonical] = {
                    "operator": op,
                    "original": name,
                    "tokens": [t.token for t in nf.tokens],
                    "case_pattern": nf.case_pattern
                }

        return self.mapping

    # store mapping
    def save_mapping(self):

        if not self.mapping or not self.level:
            raise ValueError("Generate mapping before saving it.")
        
        out_path = f"{MAPPINGS_PATH}{self.dataset}_{self.level}/{self.db_id}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, indent=4)
        
        print(f"✅ Mapping saved to {out_path}")



    #
    # recreate databases
    #

    # wrapper function to recreate databases
    def recreate_database(self):
        self.sql_create_statements()
        self.create_new_sqlite_db()
        self.copy_data()

    # create CREATE statements
    def sql_create_statements(self):

        stmts = []

        for tbl_name, tbl_info in self.schema.items():

            new_tbl_name = self.mapping[tbl_name.lower()]

            lines = []

            # pk handling
            pk_cols_old = tbl_info.get("primary_keys", [])
            pk_cols_new = [self.mapping[c.lower()] for c in pk_cols_old]
            composite_pk = len(pk_cols_old) > 1

            # columns
            for col in tbl_info["columns"]:
                old_cname = col["name"].lower()
                new_cname = self.mapping[old_cname]
                datatype = col["type"]
                notnull = " NOT NULL" if col.get("notnull") else ""

                # Only apply column-level PRIMARY KEY if it is the ONLY PK column
                if col.get("pk") and not composite_pk:
                    pk = " PRIMARY KEY"
                else:
                    pk = ""

                lines.append(f'"{new_cname}" {datatype}{notnull}{pk}')

            # composite primary key
            if composite_pk:
                pk_clause = f"PRIMARY KEY ({', '.join(pk_cols_new)})" # pk_clause = f'PRIMARY KEY ({", ".join(f\'"{c}"\' for c in pk_cols_new)})'
                lines.append(pk_clause)

            # foreign keys
            for fk in tbl_info.get("foreign_keys", []):
                
                # check if fk constraints are valid
                if any(fk.get(k) is None for k in ["sourceTable", "sourceColumn", "targetColumn"]):
                    continue
                
                src_table = self.mapping[fk["sourceTable"].lower()]
                src_col = self.mapping[fk["sourceColumn"].lower()]
                tgt_col = self.mapping[fk["targetColumn"].lower()]

                lines.append(
                    f'FOREIGN KEY ("{tgt_col}") REFERENCES "{src_table}"("{src_col}")'
                )

            # join lines with commas
            create_stmt = (
                f'CREATE TABLE "{new_tbl_name}" (\n    '
                + ',\n    '.join(lines)
                + '\n);'
            )

            stmts.append(create_stmt)

        self.create_stmts = stmts
        return self.create_stmts

    # create databases
    def create_new_sqlite_db(self):

        os.makedirs(os.path.dirname(self.db_path_new), exist_ok=True)
        if os.path.exists(self.db_path_new):
            os.remove(self.db_path_new)

        conn = sqlite3.connect(self.db_path_new)
        cur = conn.cursor()
        
        cur.execute("PRAGMA foreign_keys = OFF;") # disable FK checks while creating schema

        if not hasattr(self, "create_stmts"):
            self.sql_create_statements()

        for stmt in self.create_stmts:
            try:
                cur.execute(stmt)
            except Exception as e:
                print(stmt)
                raise Exception(e)
        
        cur.execute("PRAGMA foreign_keys = ON;") # enable FK constraints
 
        conn.commit()
        conn.close()

        print(f"Created new SQLite database at {self.db_path_new}")
    
    # copy content
    def copy_data(self):

        print(f"Copying data from {self.db_path} → {self.db_path_new}")

        old_conn = sqlite3.connect(self.db_path)
        old_conn.text_factory = bytes # return TEXT as raw bytes to avoid encoding errors
        new_conn = sqlite3.connect(self.db_path_new)

        old_cur = old_conn.cursor()
        new_cur = new_conn.cursor()

        # disable foreign key checks while inserting
        new_cur.execute("PRAGMA foreign_keys = OFF;")

        for tbl_name, tbl_info in self.schema.items():

            old_tbl = tbl_name
            new_tbl = self.mapping[tbl_name.lower()]

            # build ordered column lists (old to new)
            col_pairs = []
            for col in tbl_info["columns"]:
                old_col = col["name"]
                new_col = self.mapping[old_col.lower()]
                col_pairs.append((old_col, new_col))

            old_cols = [f'"{old}"' for old, _ in col_pairs]
            new_cols = [f'"{new}"' for _, new in col_pairs]

            select_sql = (
                f'SELECT {", ".join(old_cols)} '
                f'FROM "{old_tbl}";'
            )

            rows = old_cur.execute(select_sql).fetchall()

            if not rows:
                # print(f"[INFO] Table {old_tbl}: 0 rows (skipped)")
                continue

            placeholders = ", ".join(["?"] * len(col_pairs))

            insert_sql = (
                f'INSERT INTO "{new_tbl}" ({", ".join(new_cols)}) '
                f'VALUES ({placeholders});'
            )

            decoded_rows = [[safe_decode(v) for v in row] for row in rows]
            new_cur.executemany(insert_sql, decoded_rows)
            # print(f"[OK] Copied {len(rows)} rows into {new_tbl}")

        new_conn.commit()

        # re-enable FK checks
        new_cur.execute("PRAGMA foreign_keys = ON;")
        new_conn.commit()

        old_conn.close()
        new_conn.close()



    #
    # recreate samples (query-question pairs)
    #

    def recreate_samples(self):

        if len(self.samples) == 0: # no samples in current set
            return []

        self.dev_new = [] # reset

        for sample in self.samples:
            sql = sample.get("SQL") or sample.get("query")
            sql_new = self.translate_sql(sql=sql)

            self.dev_new.append({
                "db_id": sample["db_id"],
                "question": sample["question"],
                "query": sql_new
            })
        
        return self.dev_new

    # rewrite sql
    def translate_sql(self, sql: str) -> str:

        # parse SQL into AST
        try:
            ast = sqlglot.parse_one(sql, read="sqlite")
        except Exception as e:
            print(sql)
            raise Exception(e)

        mapping = self.mapping

        def lookup(name: str):
            # case-insensitive lookup in mapping (keys are lowercase)
            if not isinstance(name, str):
                return None
            return mapping.get(name.lower())

        def _transform(node):

            # table names
            if isinstance(node, exp.Table):
                # name is always a string table name (or None)
                old_name = node.name
                new_name = lookup(old_name)
                if new_name:
                    # set underlying identifier to the new name
                    node.set("this", exp.to_identifier(new_name))
                return node

            # column names
            if isinstance(node, exp.Column):
                # Unqualified column name
                col_name = node.name  # string or None
                new_col = lookup(col_name)
                if new_col:
                    node.set("this", exp.to_identifier(new_col))

                # table qualifier, e.g. table.column or alias.column
                # usually an identifier or sometimes a string
                tbl_expr = node.args.get("table")
                if isinstance(tbl_expr, exp.Identifier):
                    tbl_name = tbl_expr.this
                    new_tbl = lookup(tbl_name)
                    if new_tbl:
                        tbl_expr.set("this", new_tbl)
                elif isinstance(tbl_expr, str):
                    new_tbl = lookup(tbl_expr)
                    if new_tbl:
                        node.set("table", new_tbl)

                return node

            # generic identifiers
            # this catches things like stand-alone identifiers, wildcards in some contexts, etc.
            if isinstance(node, exp.Identifier):
                old = node.this
                if isinstance(old, str):
                    new = lookup(old)
                    if new:
                        node.set("this", new)
                return node

            return node

        new_ast = ast.transform(_transform)

        try:
            rewritten_sql = new_ast.sql(dialect="sqlite")            
        except Exception as e:
            print(f"[WARNING] Could not serialize AST for SQL: {sql}")
            raise Exception(e)

        return rewritten_sql


# decoding column values
def safe_decode(x):
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return x.decode("latin-1", errors="replace")
    return x