from datasets import load_dataset
import psycopg2
from pgvector.psycopg2 import register_vector

# Supabase connection string
conn_string = "postgresql://postgres.draakdgnfaofhzfijepx:SARVESH!dayma@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"

# Load FinanceBench dataset
dataset = load_dataset("PatronusAI/financebench", split="train[:100]")  # Use 100 rows for simplicity

# Connect to Supabase
conn = psycopg2.connect(conn_string)
cur = conn.cursor()
register_vector(conn)

# Create table for dense and sparse vectors
cur.execute("""
    CREATE TABLE finance_docs (
        id SERIAL PRIMARY KEY,
        text TEXT,
        dense_vector VECTOR(384),  -- Adjust size based on embedding model
        sparse_vector JSON  -- Store sparse vectors as JSON for simplicity
    );
""")

# Commit and close
conn.commit()
cur.close()
conn.close()

print("Database setup complete!")



#SARVESH!dayma