name:saumya sharma
email:saumyasharma281114@gmail.com

# hybrid-search-pgvector
This repository demonstrates how to implement hybrid search using pgvector, an extension for PostgreSQL that enables vector similarity search. It combines semantic search (using embeddings) with traditional keyword-based search to improve retrieval accuracy.


Task Chosen: Hybrid Search in PGVector


Objective: Build a hybrid search application using PGVector, compare dense vs. dense+sparse search strategies, and present the results via a frontend.


Tools: PostgreSQL with PGVector, Supabase (optional hosting), Python (backend), Athina (LLM judge), FinanceBench dataset, React (frontend).


Deliverables: GitHub repo, code walkthrough video, functional app, and analysis.

load_data.py: Sets up the Supabase database table.


embed_data.py: Loads FinanceBench data with dense and sparse vectors.


search.py: Implements dense and hybrid search logic.


app.py: Runs the Streamlit frontend.

