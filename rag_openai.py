import openai
import numpy as np
import faiss
import os
import psycopg2
import json
# FAISS Index setup
# Step 1: Chunk the source files
def chunk_source_files(folder_path, chunk_size=20):
    """
    Splits all source files in a folder into chunks.

    Args:
        folder_path (str): Path to the folder containing source files.
        chunk_size (int): Number of lines per chunk.

    Returns:
        List of chunks and a mapping of chunk to filename.
    """
    chunks = []
    chunk_mapping = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".c", ".h")):  # Handle .c and .h files
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                    for i in range(0, len(lines), chunk_size):
                        chunk = "".join(lines[i:i + chunk_size])
                        chunks.append(chunk)
                        chunk_mapping.append(file)
    return chunks, chunk_mapping


# Step 2: Embed the chunks and index them



# Function to get embeddings from OpenAI Codex (or GPT-3)
def get_code_embedding(code: str):
    """
    Retrieve code embedding using OpenAI's Codex model.

    Args:
        code (str): The code snippet for which to generate an embedding.

    Returns:
        np.array: The generated code embedding as a numpy array.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.embeddings.create(
        model="text-embedding-ada-002",  # Codex model optimized for code tasks
        input=code
    )

    # Extract the embeddings from the response
    embeddings = response.data[0].embedding
    return np.array(embeddings)


def index_chunks(chunks):
    """
    Creates embeddings and indexes the chunks using FAISS, using OpenAI Codex for code embeddings.

    Args:
        chunks (list): List of code chunks.

    Returns:
        FAISS index, embeddings.
    """
    embeddings = []

    # Generate embeddings for each code chunk using OpenAI Codex
    for chunk in chunks:
        chunk_embedding = get_code_embedding(chunk)
        embeddings.append(chunk_embedding)

    # Convert embeddings to numpy array for FAISS
    embeddings = np.array(embeddings)

    # Create FAISS index (L2 distance)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # The dimension is the size of the embeddings
    index.add(embeddings)

    return index, embeddings


def retrieve_relevant_chunks(query, chunks, index, top_k=3):
    """
    Retrieves the most relevant chunks for a given query using OpenAI Codex embeddings.

    Args:
        query (str): User's query.
        chunks (list): List of code chunks.
        index (FAISS index): Prebuilt FAISS index.
        top_k (int): Number of chunks to retrieve.

    Returns:
        List of top-k relevant chunks.
    """
    # Get the embedding of the query using OpenAI Codex
    query_embedding = get_code_embedding(query)

    # Perform similarity search with FAISS
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

    # Retrieve the most relevant code chunks based on the indices
    return [chunks[i] for i in indices[0]]

def save_faiss_index(index, file_path):
    """
    Saves the FAISS index to a file.

    Args:
        index: FAISS index object.
        file_path (str): Path to save the FAISS index.
    """
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    """
    Loads a FAISS index from a file.

    Args:
        file_path (str): Path to the FAISS index file.

    Returns:
        FAISS index object.
    """
    return faiss.read_index(file_path)



def save_to_database(embeddings, chunks, db_config):
    """
    Saves embeddings and code chunks to PostgreSQL database.

    Args:
        embeddings (numpy.ndarray): Array of embeddings.
        chunks (list): List of code chunks.
        db_config (dict): Database configuration (host, user, password, dbname).
    """
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        chunk_id = f"chunk_{i}"
        cursor.execute(
            "INSERT INTO code_embeddings (chunk_id, embedding, code_chunk) VALUES (%s, %s, %s)",
            (chunk_id, list(embedding), chunk)
        )

    conn.commit()
    cursor.close()
    conn.close()
def query_similar_embeddings(query_embedding, top_k, faiss_index, db_config):
    """
    Queries similar embeddings using FAISS and retrieves corresponding code chunks.

    Args:
        query_embedding (numpy.ndarray): Query embedding.
        top_k (int): Number of similar items to retrieve.
        faiss_index: FAISS index object.
        db_config (dict): Database configuration (host, user, password, dbname).

    Returns:
        List of similar code chunks.
    """
    # Find the nearest neighbors in FAISS
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), top_k)

    # Fetch the metadata (code chunks) for the nearest neighbors
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    similar_chunks = []
    for idx in indices[0]:
        cursor.execute("SELECT code_chunk FROM code_embeddings WHERE id = %s", (idx + 1,))
        result = cursor.fetchone()
        if result:
            similar_chunks.append(result[0])

    cursor.close()
    conn.close()

    return similar_chunks
def index_chunks_with_db(chunks, db_config, faiss_path="faiss_index.idx"):
    """
    Indexes chunks using FAISS, saves index and metadata to a database.

    Args:
        chunks (list): List of code chunks.
        db_config (dict): Database configuration.
        faiss_path (str): Path to save the FAISS index.

    Returns:
        FAISS index object.
    """
    index, embeddings = index_chunks(chunks)

    # Save the FAISS index
    save_faiss_index(index, faiss_path)

    # Save embeddings and chunks to the database
    save_to_database(embeddings, chunks, db_config)

    return index
