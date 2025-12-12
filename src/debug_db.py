import requests
import os
import pandas as pd

# Docker hostname usually 'questdb', localhost if running locally
HOST = os.getenv('QUESTDB_HOST', 'questdb')
URL = f"http://{HOST}:9000/exec"

def debug_questdb():
    print(f"🕵️‍♂️ INSPECTING QUESTDB AT: {URL}")
    print("-" * 40)

    try:
        # 1. SHOW TABLES
        # We use fmt=json to get a structured response
        r = requests.get(URL, params={'query': 'SHOW TABLES;', 'fmt': 'json'})
        
        if r.status_code != 200:
            print(f"❌ Connection Error ({r.status_code}): {r.text}")
            return

        data = r.json()
        if 'dataset' not in data:
            print("⚠️  No dataset returned from SHOW TABLES.")
            print(f"Raw Response: {data}")
            return

        tables_data = data['dataset']
        columns = [c['name'] for c in data['columns']]
        
        print(f"✅ Connection OK. Found {len(tables_data)} tables.")
        
        # Find which index holds the table name (usually 'table_name' or 'name')
        try:
            name_idx = next(i for i, c in enumerate(columns) if c in ['table_name', 'name'])
        except StopIteration:
            name_idx = 1 # Default fallback
            print(f"⚠️  Could not find 'name' column in {columns}. Guessing index 1.")

        # 2. ITERATE TABLES
        for row in tables_data:
            table_name = row[name_idx]
            
            # Check Row Count
            count_q = f"SELECT count() FROM '{table_name}'"
            cr = requests.get(URL, params={'query': count_q, 'fmt': 'json'})
            
            if cr.status_code == 200:
                row_count = cr.json()['dataset'][0][0]
            else:
                row_count = "Error"

            print(f"\n📁 TABLE: {table_name}")
            print(f"   Rows: {row_count}")

            # Check Columns (Schema)
            if isinstance(row_count, int) and row_count >= 0:
                schema_q = f"SELECT * FROM '{table_name}' LIMIT 1"
                sr = requests.get(URL, params={'query': schema_q, 'fmt': 'json'})
                if sr.status_code == 200:
                    cols = [c['name'] for c in sr.json()['columns']]
                    print(f"   Cols: {cols}")

    except Exception as e:
        print(f"❌ CRITICAL SCRIPT ERROR: {e}")

if __name__ == "__main__":
    debug_questdb()