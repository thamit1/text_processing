# Import necessary libraries
import numpy as np
from transformers import BertTokenizer, TFBertModel
import faiss

# Load the model and tokenizer
model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to encode text into embeddings
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='tf', padding='max_length', truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Use the CLS token
    return embeddings.numpy()

# Split text into smaller chunks with overlap
def split_into_chunks(text, max_length, stride):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), stride)]
    return [" ".join(chunk) for chunk in chunks]

# Example documents
documents = [
    "Document 1 text, which is quite large and contains a lot of information.",
    "Document 2 text, another long document with important details.",
    "Document 3 text, yet another lengthy document for testing."
]

# Encode and store embeddings
doc_embeddings = []
doc_map = []

for doc_id, doc in enumerate(documents):
    chunks = split_into_chunks(doc, max_length=512, stride=256)
    for chunk_id, chunk in enumerate(chunks):
        embedding = encode_text(chunk, tokenizer, model)
        doc_embeddings.append(embedding)
        doc_map.append((doc_id, chunk_id))

# Create FAISS index
dimension = doc_embeddings[0].shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.vstack(doc_embeddings))
faiss.write_index(index, 'faiss_chunks_index.index')

# Function to search for the most relevant document
def search_document(query, tokenizer, model, index, doc_map, documents):
    query_embedding = encode_text(query, tokenizer, model)
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 nearest neighbors
    
    results = {}
    for i in I[0]:
        doc_id, chunk_id = doc_map[i]
        if doc_id not in results:
            results[doc_id] = 0
        results[doc_id] += 1  # Aggregate scores based on chunks
    
    best_doc_id = max(results, key=results.get)
    return documents[best_doc_id], results[best_doc_id]

# Example query
query = "specific issue in ticket"
index = faiss.read_index('faiss_chunks_index.index')

# Search for the most relevant document
best_doc, score = search_document(query, tokenizer, model, index, doc_map, documents)
print(f"Best matching document: {best_doc}")
print(f"Score: {score}")
