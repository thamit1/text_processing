from flask import Flask, render_template, request
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re

# Configure logging with timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# In-memory cache
cache = {}

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')

# Load stored data
with open('ticket_summaries.json', 'r') as f:
    ticket_data_with_summaries = json.load(f)

stored_embeddings = np.load('ticket_embeddings_v2.npy', allow_pickle=True)
ticket_descriptions = [ticket['description'] for ticket in ticket_data_with_summaries]

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_query = request.form['msg']
    # Check if the response is in the cache
    if user_query in cache:
        response = cache[user_query]
        logging.info("Cache hit")
    else:
        response = handle_user_query(user_query)
        # Store the response in the cache
        cache[user_query] = response
        logging.info("Cache miss")
    return str(response)

def handle_user_query(query):
    try:
        # Normalize the embeddings
        def normalize_embeddings(embeddings):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms

        # Check if the query contains a ticket number
        ticket_no_pattern = re.compile(r'\bQBC-\d{4}\b')
        ticket_no_match = ticket_no_pattern.search(query)

        if ticket_no_match:
            # If a ticket number is mentioned in the query, find the corresponding ticket
            ticket_no = ticket_no_match.group()
            specific_ticket = next((ticket for ticket in ticket_data_with_summaries if ticket['ticketNo'] == ticket_no), None)
            if specific_ticket:
                specific_ticket_chunks = chunk_text(specific_ticket['description'])
                specific_ticket_desc_embeddings = normalize_embeddings(embedding_model.encode(specific_ticket_chunks))
                similarities = cosine_similarity(specific_ticket_desc_embeddings, normalize_embeddings(stored_embeddings))

                # Aggregate the similarities for all chunks
                aggregated_similarities = np.mean(similarities, axis=0)

                logging.info(f"Aggregated similarities length: {len(aggregated_similarities)}")
                logging.info(f"Ticket data length: {len(ticket_data_with_summaries)}")

                # Check if lengths match
                if len(aggregated_similarities) != len(ticket_data_with_summaries):
                    logging.error("Length mismatch between aggregated similarities and ticket data")
                    return "An error occurred while processing your request."
                
                # Find all tickets that are similar to the specific ticket description/summary
                similar_tickets = [(ticket_data_with_summaries[i], similarity) for i, similarity in enumerate(aggregated_similarities) if similarity > 0.3]
        else:
            # Otherwise, find similar tickets based on query embedding
            query_chunks = chunk_text(query)
            query_embedding = normalize_embeddings(embedding_model.encode(query_chunks))
            similarities = cosine_similarity(query_embedding, normalize_embeddings(stored_embeddings))

            # Aggregate the similarities for all chunks
            aggregated_similarities = np.mean(similarities, axis=0)

            logging.info(f"Aggregated similarities length: {len(aggregated_similarities)}")
            logging.info(f"Ticket data length: {len(ticket_data_with_summaries)}")

            # Check if lengths match
            if len(aggregated_similarities) != len(ticket_data_with_summaries):
                logging.error("Length mismatch between aggregated similarities and ticket data")
                return "An error occurred while processing your request."
            
            # Find all tickets that are similar
            similar_tickets = [(ticket_data_with_summaries[i], similarity) for i, similarity in enumerate(aggregated_similarities) if similarity > 0.26]

        # Sort tickets by similarity in descending order
        similar_tickets.sort(key=lambda x: x[1], reverse=True)

        # Create a response for all similar tickets
        responses = []
        for ticket, similarity in similar_tickets:
            summary = ticket['summary']
            ticket_no = ticket['ticketNo']
            responses.append(f"<b>{ticket_no} - </b> {summary} (Similarity: {similarity:.2f})")

        response_text = "<br>".join(responses)
        # Generate a response without unrelated details
        response = f'<p style="color: blue; font-style: italic;">User query: {query}</p>{response_text}'
        return response
    except Exception as e:
        logging.error(f"Error in handle_user_query: {e}")
        return "An error occurred while processing your request."

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

if __name__ == '__main__':
    app.run(debug=True)
