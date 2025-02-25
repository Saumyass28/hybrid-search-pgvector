from datasets import load_dataset
import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Load model for dense vectors
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Supabase connection
conn_string = "postgresql://postgres.draakdgnfaofhzfijepx:SARVESH!dayma@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()
register_vector(conn)

# Load dataset
dataset = load_dataset("PatronusAI/financebench", split="train[:100]")

# Function to generate dense vectors
def get_dense_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

# Function to generate sparse vectors (simple TF-IDF-like approach)
def get_sparse_vector(text):
    words = text.split()
    word_counts = {word: words.count(word) for word in set(words)}
    return json.dumps(word_counts)  # Store as JSON

# Insert data with embeddings
for row in dataset:
    text = row["question"] + " " + row["answer"]  # Combine question and answer
    dense_vec = get_dense_vector(text)
    sparse_vec = get_sparse_vector(text)
    cur.execute(
        "INSERT INTO finance_docs (text, dense_vector, sparse_vector) VALUES (%s, %s, %s)",
        (text, dense_vec.tolist(), sparse_vec)
    )

conn.commit()
cur.close()
conn.close()
print("Data loaded with embeddings!")