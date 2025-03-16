import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
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


def index_chunks(chunks):
    """
    Creates embeddings and indexes the chunks using FAISS, using GraphCodeBERT for code embeddings.
    Args:
        chunks (list): List of code chunks.
    Returns:
        FAISS index, embeddings, and the GraphCodeBERT model.
    """
    # Load GraphCodeBERT model and tokenizer
    model_name = 'microsoft/codebert-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)  # Get all hidden states
    model = model.to("cpu")

    # Tokenize and generate embeddings for each chunk of code
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():  # Disable gradients during inference
            outputs = model(**inputs)
        # We use all hidden states to generate embeddings (avoid the pooler output)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling over the token embeddings
        embeddings.append(chunk_embedding)

    # Convert embeddings to numpy array for FAISS
    embeddings = np.array(embeddings)

    # Create FAISS index (L2 distance)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # The dimension is the size of the embeddings
    index.add(embeddings)

    return index, embeddings, tokenizer, model




# Step 3: Retrieve relevant chunks
def retrieve_relevant_chunks(query, chunks, index, tokenizer, model, top_k=3):
    """
    Retrieves the most relevant chunks for a given query using GraphCodeBERT embeddings.

    Args:
        query (str): User's query.
        chunks (list): List of code chunks.
        index (FAISS index): Prebuilt FAISS index.
        tokenizer (RobertaTokenizer): Tokenizer for GraphCodeBERT.
        model (RobertaModel): GraphCodeBERT model.
        top_k (int): Number of chunks to retrieve.

    Returns:
        List of top-k relevant chunks.
    """
    # Tokenize the query and generate its embedding using GraphCodeBERT

    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling to get the query embedding
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().astype(np.float32)

    # Perform similarity search with FAISS
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

    # Retrieve the most relevant code chunks based on the indices
    return [chunks[i] for i in indices[0]]
