from flask import Flask, render_template, request
import logging
import time
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Configure logging with timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# In-memory cache
cache = {}

# Load models
summarization_model_name = 'facebook/bart-large-cnn'
summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name, cache_dir='./model_cache')
summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name, cache_dir='./model_cache')

response_model_name = 'gpt2'
response_tokenizer = GPT2Tokenizer.from_pretrained(response_model_name, cache_dir='./model_cache')
response_model = GPT2LMHeadModel.from_pretrained(response_model_name, cache_dir='./model_cache')

# Explicitly set the pad_token_id for GPT-2
response_tokenizer.pad_token_id = response_tokenizer.eos_token_id

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
        query_embedding = embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, stored_embeddings)
        most_similar_index = np.argmax(similarities)
        most_similar_ticket = ticket_data_with_summaries[most_similar_index]

        summary = summarize_text(most_similar_ticket['description'])
        response = generate_response(f"User query: {query}\nSummary: {summary}")
        return response
    except Exception as e:
        logging.error(f"Error in handle_user_query: {e}")
        return "An error occurred while processing your request."

def summarize_text(text, max_length=80, min_length=40):
    try:
        start_time = time.time()
        logging.info("Summarize function called")
        inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarization_model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        end_time = time.time()
        logging.info(f"Summarize function completed in {end_time - start_time} seconds")
        return summary
    except Exception as e:
        logging.error(f"Error in summarize_text: {e}")
        return "An error occurred while summarizing the text."

def generate_response(prompt, max_new_tokens=80):
    try:
        start_time = time.time()
        logging.info("Generate response function called")
        inputs = response_tokenizer.encode(prompt, return_tensors='pt')
        logging.info(f"Input length: {len(inputs[0])}")
        attention_mask = inputs.ne(response_tokenizer.pad_token_id).long()
        logging.info(f"Attention mask length: {len(attention_mask[0])}")

        logging.info("Using non-quantized model")
        outputs = response_model.generate(
            inputs, 
            max_new_tokens=max_new_tokens, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            early_stopping=True, 
            attention_mask=attention_mask,
            pad_token_id=response_tokenizer.pad_token_id
        )

        logging.info(f"Output length: {len(outputs[0])}")
        response = response_tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Response: {response}")

        # Ensure the response is complete by checking for the end of sentence punctuation
        sentences = response.split('. ')
        if sentences[-1] and not sentences[-1].endswith(('.', '!', '?')):
            sentences = sentences[:-1]
        complete_response = '. '.join(sentences)
        logging.info(f"Complete response: {complete_response}")
        end_time = time.time()
        logging.info(f"Generate response function completed in {end_time - start_time} seconds")
        return complete_response
    except Exception as e:
        logging.error(f"Error in generate_response: {e}")
        return "An error occurred while generating the response."

if __name__ == '__main__':
    app.run(debug=True)
