# ---- sqlite monkeypatch (fix before chroma import) ----
import sys
try:
    import pysqlite3 as sqlite3_new
    sys.modules["sqlite3"] = sqlite3_new
    print("✅ Using pysqlite3 (SQLite version):", sqlite3_new.sqlite_version)
except Exception as e:
    print("⚠️ Could not load pysqlite3:", e)

# ---- normal imports ----
import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ------------------------------
# Setup Embedding Model
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()

# ------------------------------
# Setup ChromaDB Persistent Client
# ------------------------------
DB_DIR = "chroma_ads_db"

@st.cache_resource
def get_chroma_collection():
    client = PersistentClient(path=DB_DIR)
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
st.set_page_config(page_title="Ad Retriever - v2 ", page_icon="📢", layout="centered")

st.title("📢 Ad Retriever (RAG Demo)")
st.write("Type your query below and get the most relevant ads.")

# Load and index ads (only once)
ads = load_ads("ads.txt")
if ads:
    index_ads(ads)

# 🚀 The Input Box (should now appear)
user_query = st.text_input("Enter your query:")

top_k = st.slider("Number of ads to retrieve:", 1, 10, 5)

if user_query:
    results = retrieve_ads(user_query, top_k=top_k)
    st.write("### 🔎 Top Relevant Ads")
    for ad in results:
        st.write(f"- {ad}")
