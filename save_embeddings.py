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
    2. Run the script to generate and save the normalized embeddings to 'ticket_embeddings.npy'.
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
    ticket_descriptions = json.load(file)

# Generate embeddings
embeddings = embedding_model.encode(ticket_descriptions)

# Normalize embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

normalized_embeddings = normalize_embeddings(embeddings)

# Save normalized embeddings to a file
np.save('ticket_embeddings.npy', normalized_embeddings)
