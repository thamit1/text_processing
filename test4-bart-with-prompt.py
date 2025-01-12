from transformers import BartTokenizer, BartForConditionalGeneration
import time
import logging
"""
This script uses the BART model from the Hugging Face Transformers library to summarize text with important points included in the prompt.
Functions:
    summarize_text(text: str, important_points: list, max_length: int = 150, min_length: int = 40) -> str:
        Summarizes the given text, incorporating the specified important points.
Parameters:
    text (str): The text to be summarized.
    important_points (list): A list of important points to be included in the summary prompt.
    max_length (int, optional): The maximum length of the summary. Defaults to 150.
    min_length (int, optional): The minimum length of the summary. Defaults to 40.
Returns:
    str: The summarized text or an error message if an exception occurs.
Example usage:
"""

# Load models
summarization_model_name = 'facebook/bart-large-cnn'
summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name, cache_dir='./model_cache')
summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name, cache_dir='./model_cache')

def summarize_text(text, important_points, max_length=150, min_length=40):
    try:
        start_time = time.time()
        logging.info("Summarize function called")

        # Include important points in the text prompt
        prompt = "summarize: " + text + " Important points: " + ", ".join(important_points)
        inputs = summarization_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarization_model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        end_time = time.time()
        logging.info(f"Summarize function completed in {end_time - start_time} seconds")
        return summary
    except Exception as e:
        logging.error(f"Error in summarize_text: {e}")
        return "An error occurred while summarizing the text."

# Example usage
text = "Your input text here"
important_points = ["Point 1", "Point 2", "Point 3"]
summary = summarize_text(text, important_points)
print(summary)
