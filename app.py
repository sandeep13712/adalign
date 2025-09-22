import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# ------------------------------
# Setup Embedding Model
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()

# ------------------------------
# Setup ChromaDB (Persistent)
# ------------------------------
DB_DIR = "chroma_ads_db"

@st.cache_resource
def get_chroma_collection():
    client = chromadb.Client(persist_directory=DB_DIR)
    return client.get_or_create_collection("ads_collection")

collection = get_chroma_collection()

# ------------------------------
# Load Ads from File
# ------------------------------
def load_ads(file_path="ads.txt"):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        ads = [line.strip() for line in f if line.strip()]
    return ads

def index_ads(ads):
    # Only add new ads not already in DB
    existing_ids = set(collection.get()["ids"])
    for i, ad in enumerate(ads):
        ad_id = str(i)
        if ad_id not in existing_ids:
            embedding = embedder.encode([ad], convert_to_numpy=True)[0]
            collection.add(
                documents=[ad],
                embeddings=[embedding],
                metadatas=[{"ad_id": ad_id}],
                ids=[ad_id],
            )
    # Persist changes
    collection.client.persist()

# ------------------------------
# Query Function
# ------------------------------
def retrieve_ads(user_query, top_k=5):
    query_embedding = embedder.encode([user_query], convert_to_numpy=True)
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title(" Ad Retriever (RAG Demo)")
st.write("Type your query below and get the most relevant ads.")

# Load and index ads (runs only once)
ads = load_ads("ads.txt")
if ads:
    index_ads(ads)

# User Query
user_query = st.text_input("Enter your query:")
top_k = st.slider("Number of ads to retrieve:", 1, 10, 5)

if user_query:
    results = retrieve_ads(user_query, top_k=top_k)
    st.write("### 🔎 Top Relevant Ads")
    for ad in results:
        st.write(f"- {ad}")
