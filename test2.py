from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('paraphrase-mpnet-base-v2', cache_folder='./model_cache')

# Example sentences
sentences = ["This is an example sentence", "Each sentence is converted"]

# Generate embeddings
embeddings = model.encode(sentences)
print(embeddings)

# Load another model
model = SentenceTransformer('all-mpnet-base-v2', cache_folder='./model_cache')
# Generate embeddings
embeddings = model.encode(sentences)
print(embeddings)

