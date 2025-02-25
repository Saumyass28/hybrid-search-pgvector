import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Load model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Supabase connection
conn_string = "postgresql://postgres.draakdgnfaofhzfijepx:SARVESH!dayma@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"

def get_dense_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

def get_sparse_vector(text):
    words = text.split()
    return json.dumps({word: words.count(word) for word in set(words)})

# Dense-only search
def dense_search(query, limit=5):
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    register_vector(conn)
    query_vec = get_dense_vector(query)
    cur.execute(
        "SELECT id, text, dense_vector <-> %s::vector AS distance FROM finance_docs ORDER BY distance LIMIT %s",
        (query_vec.tolist(), limit)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [{"id": r[0], "text": r[1], "distance": r[2]} for r in results]

# Dense + sparse hybrid search
def hybrid_search(query, limit=5):
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    register_vector(conn)
    query_vec = get_dense_vector(query)
    query_sparse = json.loads(get_sparse_vector(query))  # Convert query JSON string to dict

    # Get dense distances and sparse vectors from the database
    cur.execute(
        "SELECT id, text, dense_vector <-> %s::vector AS distance, sparse_vector FROM finance_docs ORDER BY distance LIMIT %s",
        (query_vec.tolist(), limit)
    )
    results = cur.fetchall()

    # Compute hybrid scores in Python
    hybrid_results = []
    for result in results:
        id, text, distance, sparse_vector = result
        sparse_dict = sparse_vector  # Already a dict from psycopg2

        # Simple overlap similarity for sparse vectors
        query_keys = set(query_sparse.keys())
        doc_keys = set(sparse_dict.keys())
        overlap = len(query_keys & doc_keys) / max(len(query_keys), len(doc_keys)) if max(len(query_keys), len(doc_keys)) > 0 else 0
        sparse_score = 1 - overlap  # Convert to "distance" (lower is better)

        # Combine dense and sparse scores
        hybrid_score = distance + sparse_score
        hybrid_results.append({"id": id, "text": text, "score": hybrid_score})

    cur.close()
    conn.close()
    return sorted(hybrid_results, key=lambda x: x["score"])[:limit]  # Sort and limit in Python

if __name__ == "__main__":
    query = "What is the financial performance of this company?"
    print("Dense Search Results:")
    for res in dense_search(query):
        print(res)
    print("\nHybrid Search Results:")
    for res in hybrid_search(query):
        print(res)