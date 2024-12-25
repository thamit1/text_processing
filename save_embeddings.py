from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')

# List of ticket descriptions (replace with your actual descriptions)
# Load ticket descriptions from JSON file
with open('ticket_summaries.json', 'r') as file:
    ticket_descriptions = json.load(file)

# Generate embeddings
embeddings = embedding_model.encode(ticket_descriptions)

# Save embeddings to a file
np.save('ticket_embeddings.npy', embeddings)
