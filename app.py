import streamlit as st
import os

# ✅ Import functions from rag_flow.py
from rag_flow import (
    load_chunk_documents,
    VectorDB,
    query_document,
    MongoLogger
)

st.set_page_config(page_title="📄 RAG Document Q&A", layout="wide")
st.title("📄 RAG Document Q&A with NVIDIA + ChromaDB")

logger = MongoLogger()

# --- Maintain session state for file ---
if "file_path" not in st.session_state:
    st.session_state.file_path = None

# --- Step 1: Upload & Process ---
st.subheader("Step 1: Upload & Index Document")
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.info(f"✅ Uploaded: {uploaded_file.name}")
    os.makedirs("Documents", exist_ok=True)

    # Save temporarily
    temp_path = os.path.join("Documents", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Process & Index Document"):
        with st.spinner("⏳ Processing document into chunks & indexing..."):
            docs = load_chunk_documents(temp_path)
            vector_store, collection_name = VectorDB(docs, temp_path)
            st.session_state.file_path = temp_path  # Save for later queries
            st.success(f"✅ Document processed & stored as `{collection_name}`")
            st.balloons()

# --- Step 2: Query ---
st.subheader("Step 2: Ask Questions")
if st.session_state.file_path:
    st.success(f"📄 Ready! You can now query `{os.path.basename(st.session_state.file_path)}`")
    question = st.text_input("Enter your question about this document")

    if question and st.button("Get Answer"):
        with st.spinner("🤖 Querying document..."):
            answer, retrieved_context = query_document(st.session_state.file_path, question, logger)
        st.subheader("✅ Answer:")
        st.success(answer)

        with st.expander("🔍 Retrieved Context"):
            st.text_area("Context", retrieved_context, height=200)

else:
    st.warning("⚠️ Please upload & process a document first.")
