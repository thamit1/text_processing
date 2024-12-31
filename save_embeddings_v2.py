from sentence_transformers import SentenceTransformer
import numpy as np
import json

"""
This script loads a list of ticket descriptions from a JSON file, generates embeddings for each description using a pre-trained SentenceTransformer model, normalizes the embeddings, and saves the embeddings to a NumPy file.
Modules:
    - sentence_transformers: Provides the SentenceTransformer class for generating embeddings.
    - numpy: Used for saving the embeddings as a NumPy file.
    - json: Used for loading the ticket descriptions from a JSON file.
Functions:
    None
Usage:
    1. Ensure you have a JSON file named 'ticket_summaries.json' containing the ticket descriptions.
    2. Run the script to generate and save the normalized embeddings to 'ticket_embeddings_v2.npy'.
Example:
    $ python save_embeddings.py
Dependencies:
    - sentence-transformers
    - numpy
    - json
Notes:
    - The SentenceTransformer model 'all-MiniLM-L6-v2' is used for generating embeddings.
    - The model is cached in the './model_cache' directory.
"""

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')

# Load ticket descriptions from JSON file
with open('ticket_summaries.json', 'r') as file:
    ticket_data_with_summaries = json.load(file)

ticket_descriptions = [ticket['description'] for ticket in ticket_data_with_summaries]

# Function to chunk text into smaller parts
def chunk_text(text, max_length=100):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Generate embeddings for chunks and aggregate them
aggregated_embeddings = []
for description in ticket_descriptions:
    chunks = chunk_text(description)
    chunk_embeddings = embedding_model.encode(chunks)
    
    # Aggregate the chunk embeddings for each ticket by taking the mean
    aggregated_embedding = np.mean(chunk_embeddings, axis=0)
    aggregated_embeddings.append(aggregated_embedding)

# Normalize embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

normalized_embeddings = normalize_embeddings(np.array(aggregated_embeddings))

# Save normalized embeddings to a file
np.save('ticket_embeddings_v2.npy', normalized_embeddings)
