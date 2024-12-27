from flask import Flask, render_template, request
import logging
# import time
# import torch
# from transformers import BartTokenizer, BartForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
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
# summarization_model_name = 'facebook/bart-large-cnn'
# summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name, cache_dir='./model_cache')
# summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name, cache_dir='./model_cache')

# response_model_name = 'gpt2'
# response_tokenizer = GPT2Tokenizer.from_pretrained(response_model_name, cache_dir='./model_cache')
# response_model = GPT2LMHeadModel.from_pretrained(response_model_name, cache_dir='./model_cache')

# # Explicitly set the pad_token_id for GPT-2
# response_tokenizer.pad_token_id = response_tokenizer.eos_token_id

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')

# Load stored data
with open('ticket_summaries.json', 'r') as f:
    ticket_data_with_summaries = json.load(f)

stored_embeddings = np.load('ticket_embeddings.npy', allow_pickle=True)
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
                specific_ticket_desc_embedding = normalize_embeddings(embedding_model.encode([specific_ticket['description']]))
                similarities = cosine_similarity(specific_ticket_desc_embedding, normalize_embeddings(stored_embeddings))

                # Find all tickets that are similar to the specific ticket description/summary
                similar_tickets = [(ticket_data_with_summaries[i], similarity) for i, similarity in enumerate(similarities[0]) if similarity > 0.3]
        else:
            # Otherwise, find similar tickets based on query embedding
            query_embedding = normalize_embeddings(embedding_model.encode([query]))
            similarities = cosine_similarity(query_embedding, normalize_embeddings(stored_embeddings))

            # Find all tickets that are similar
            similar_tickets = [(ticket_data_with_summaries[i], similarity) for i, similarity in enumerate(similarities[0]) if similarity > 0.26]

        # Sort tickets by similarity in descending order
        similar_tickets.sort(key=lambda x: x[1], reverse=True)

        # Create a summarized response for all similar tickets
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

# def summarize_text(text, max_length=80, min_length=40):
#     try:
#         start_time = time.time()
#         logging.info("Summarize function called")
#         inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
#         summary_ids = summarization_model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
#         summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         end_time = time.time()
#         logging.info(f"Summarize function completed in {end_time - start_time} seconds")
#         return summary
#     except Exception as e:
#         logging.error(f"Error in summarize_text: {e}")
#         return "An error occurred while summarizing the text."

if __name__ == '__main__':
    app.run(debug=True)
