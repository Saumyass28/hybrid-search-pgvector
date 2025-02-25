import streamlit as st
from search import dense_search, hybrid_search

st.title("Hybrid Search Demo")

query = st.text_input("Enter your search query:", "What is the financial performance of this company?")
if st.button("Search"):
    st.subheader("Dense Search Results")
    dense_results = dense_search(query)
    for res in dense_results:
        st.write(f"Text: {res['text'][:200]}... | Distance: {res['distance']:.4f}")

    st.subheader("Hybrid Search Results")
    hybrid_results = hybrid_search(query)
    for res in hybrid_results:
        st.write(f"Text: {res['text'][:200]}... | Score: {res['score']:.4f}")