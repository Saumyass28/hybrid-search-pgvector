from search import dense_search, hybrid_search
from datasets import load_dataset

# Load 20 sample queries
dataset = load_dataset("PatronusAI/financebench", split="train[:20]")
queries = [row["question"] for row in dataset]

# Run comparisons
dense_results = {}
hybrid_results = {}
for query in queries:
    dense_results[query] = dense_search(query)
    hybrid_results[query] = hybrid_search(query)

# Simple analysis (manual relevance check or use LLM judge like Athina)
print("Sample Comparison:")
for query in queries[:5]:
    print(f"\nQuery: {query}")
    print("Dense:", dense_results[query][0]["text"][:100], "...")
    print("Hybrid:", hybrid_results[query][0]["text"][:100], "...")