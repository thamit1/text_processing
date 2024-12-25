### 1. Imports and Configuration

```python
from flask import Flask, render_template, request
import logging
import time
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
```

- **Imports**: Import necessary libraries for Flask, logging, time handling, and various machine learning models and utilities.
- **Logging Configuration**: Set up logging to capture events and time-stamped messages at the INFO level.

### 2. Flask App Initialization

```python
app = Flask(__name__)
```

- **Initialize Flask App**: Create a Flask application instance.

### 3. In-Memory Cache

```python
cache = {}
```

- **Cache Initialization**: Create an in-memory cache (a Python dictionary) to store responses.

### 4. Model Loading

```python
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
```

- **Summarization Model**: Load the BART model and tokenizer for text summarization.
- **Response Model**: Load the GPT-2 model and tokenizer for generating responses.
- **Pad Token ID**: Explicitly set the pad token ID for the GPT-2 model.
- **Embedding Model**: Load the SentenceTransformer model for generating text embeddings.

### 5. Load Stored Data

```python
# Load stored data
with open('ticket_summaries.json', 'r') as f:
    ticket_data_with_summaries = json.load(f)

stored_embeddings = np.load('ticket_embeddings.npy', allow_pickle=True)
ticket_descriptions = [ticket['description'] for ticket in ticket_data_with_summaries]
```

- **Load Data**: Load precomputed summaries and embeddings from JSON and NumPy files.

### 6. Flask Routes

```python
@app.route('/')
def home():
    return render_template('chat.html')
```

- **Home Route**: Render the homepage (chat interface).

```python
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
```

- **Chatbot Response Route**: Handle POST requests from the chat interface.
  - **Cache Check**: Check if the response is already in the cache.
  - **Cache Miss**: If not in cache, process the query and store the result.

### 7. Query Handling

```python
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
```

- **handle_user_query Function**: Process user queries.
  - **Encode Query**: Generate embeddings for the user query.
  - **Find Similarities**: Calculate cosine similarities between the query and stored embeddings.
  - **Find Most Similar**: Identify the most similar stored ticket.
  - **Summarize**: Summarize the most similar ticket description.
  - **Generate Response**: Generate a response based on the query and summary.

### 8. Summarize Text

```python
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
```

- **summarize_text Function**: Summarize a given text.
  - **Tokenize Input**: Encode the input text.
  - **Generate Summary**: Use the BART model to generate a summary.
  - **Decode Summary**: Decode the generated summary.

### 9. Generate Response

```python
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
```

- **generate_response Function**: Generate a response based on the input prompt.
  - **Tokenize Input**: Encode the prompt.
  - **Generate Output**: Use the GPT-2 model to generate a response.
  - **Decode Output**: Decode the generated response.
  - **Ensure Completeness**: Ensure the response ends with proper punctuation.

### 10. Run Flask App

```python
if __name__ == '__main__':
    app.run(debug=True)
```

- **Run Flask App**: Start the Flask application in debug mode.


## Why do we need to incorporate both, json and embeddings?
### 1. Loading `ticket_summaries.json`:
```python
with open('ticket_summaries.json', 'r') as f:
    ticket_data_with_summaries = json.load(f)
```
- **Purpose**: This code reads the `ticket_summaries.json` file and loads the JSON data into a Python list of dictionaries. Each dictionary contains descriptions and summaries of tickets.
- **Usage**: The loaded data is used to provide context and relevant information about the tickets. This data can be used for tasks such as summarization, generating responses, and finding similarities.

### 2. Loading `ticket_embeddings.npy`:
```python
stored_embeddings = np.load('ticket_embeddings.npy', allow_pickle=True)
```
- **Purpose**: This code reads the `ticket_embeddings.npy` file and loads the precomputed embeddings (numerical representations) of the ticket descriptions into a NumPy array.
- **Usage**: The embeddings are used to efficiently compute similarities between user queries and stored ticket descriptions. This allows the application to find and return the most relevant ticket based on the query.

#### 3. Extracting `ticket_descriptions`:
```python
ticket_descriptions = [ticket['description'] for ticket in ticket_data_with_summaries]
```
- **Purpose**: This code extracts the descriptions from the loaded ticket data and stores them in a list.
- **Usage**: While this step might seem redundant given the other data structures, it's useful for quickly accessing just the descriptions for tasks that might need them separately.

#### Summary of How They Work Together:
1. **Context**: `ticket_data_with_summaries` provides the full context of each ticket, including descriptions and summaries.
2. **Similarity Search**: `stored_embeddings` allows for efficient similarity searches to find tickets relevant to a user's query.
3. **Description Access**: `ticket_descriptions` provides easy access to just the descriptions, which might be handy for certain operations.

In summary, loading both the ticket summaries and embeddings allows your application to handle both the retrieval of context-rich information and the efficient computation of similarities, improving the overall functionality and user experience.