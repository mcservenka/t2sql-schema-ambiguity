import sqlite3
from func_timeout import func_timeout, FunctionTimedOut

# testing samples
def verify_sample(sql: str, db_path: str):

    def run_query():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            cur.execute(sql)
            cur.fetchone()     # optional: forces execution
        finally:
            conn.close()

    try:
        func_timeout(30, run_query)
        return True
    except FunctionTimedOut:
        print("Timeout:", sql)
        return False
    except Exception as e:
        print(db_path, ":", sql)
        print(e)
        print("------------------------------------")
        return False
