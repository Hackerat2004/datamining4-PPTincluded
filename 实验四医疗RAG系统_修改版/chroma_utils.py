import streamlit as st
import chromadb
from chromadb.config import Settings
import time
import os

from config import (
    CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, TOP_K, id_to_doc_map
)

@st.cache_resource
def get_chroma_client():
    """Initializes and returns a ChromaDB client."""
    try:
        st.write(f"Initializing ChromaDB client with persist directory: {CHROMA_PERSIST_DIR}")
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        st.success("ChromaDB client initialized!")
        return client
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB client: {e}")
        return None

@st.cache_resource
def setup_chroma_collection(_client):
    """Creates or loads the ChromaDB collection."""
    try:
        collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        st.write(f"Collection '{COLLECTION_NAME}' is ready.")
        return collection
    except Exception as e:
        st.error(f"Error setting up ChromaDB collection '{COLLECTION_NAME}': {e}")
        return None

def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it using ChromaDB."""
    global id_to_doc_map

    if not client:
        st.error("ChromaDB client not available for indexing.")
        return False

    collection = setup_chroma_collection(client)
    if not collection:
        return False

    try:
        current_count = collection.count()
    except Exception:
        current_count = 0

    st.write(f"Documents currently in ChromaDB collection '{COLLECTION_NAME}': {current_count}")

    data_to_index = data[:MAX_ARTICLES_TO_INDEX]
    needed_count = 0
    docs_for_embedding = []
    ids_to_insert = []
    temp_id_map = {}

    with st.spinner("Preparing data for indexing..."):
        for i, doc in enumerate(data_to_index):
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""
            content = f"Title: {title}\nAbstract: {abstract}".strip()
            if not content:
                continue

            doc_id = str(i)
            needed_count += 1
            temp_id_map[i] = {
                'title': title, 
                'abstract': abstract, 
                'content': content,
                'source_file': doc.get('source_file')
            }
            docs_for_embedding.append(content)
            ids_to_insert.append(doc_id)

    if current_count < needed_count and docs_for_embedding:
        st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")

        st.write(f"Embedding {len(docs_for_embedding)} documents...")
        with st.spinner("Generating embeddings..."):
            start_embed = time.time()
            embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
            end_embed = time.time()
            st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")

        st.write("Inserting data into ChromaDB...")
        with st.spinner("Inserting..."):
            try:
                start_insert = time.time()
                collection.add(
                    ids=ids_to_insert,
                    embeddings=embeddings.tolist(),
                    metadatas=[{"source": doc.get('source_file', 'unknown')} for doc in data_to_index],
                    documents=docs_for_embedding
                )
                end_insert = time.time()
                inserted_count = len(ids_to_insert)
                st.success(f"Successfully indexed {inserted_count} documents. Insert took {end_insert - start_insert:.2f} seconds.")
                id_to_doc_map.update(temp_id_map)
                return True
            except Exception as e:
                st.error(f"Error inserting data into ChromaDB: {e}")
                return False
    elif current_count >= needed_count:
        st.write("Data count suggests indexing is complete.")
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True
    else:
        st.error("No valid text content found in the data to index.")
        return False

def search_similar_documents(client, query, embedding_model):
    """Searches ChromaDB for documents similar to the query."""
    if not client or not embedding_model:
        st.error("ChromaDB client or embedding model not available for search.")
        return [], []

    collection = setup_chroma_collection(client)
    if not collection:
        return [], []

    try:
        query_embedding = embedding_model.encode([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=TOP_K,
            include=["distances", "documents", "metadatas"]
        )

        if not results or not results['ids'] or not results['ids'][0]:
            return [], []

        hit_ids = [int(id_str) for id_str in results['ids'][0]]
        distances = results['distances'][0]

        return hit_ids, distances
    except Exception as e:
        st.error(f"Error during ChromaDB search: {e}")
        return [], []
