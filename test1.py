# This script demonstrates how to load a pre-trained GPT-2 model 
# and generate a response to a prompt using the model.
import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Configure logging with timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and tokenizer
response_model_name = 'gpt2'
response_tokenizer = GPT2Tokenizer.from_pretrained(response_model_name, cache_dir='./model_cache')
response_model = GPT2LMHeadModel.from_pretrained(response_model_name, cache_dir='./model_cache')

# Quantize the GPT-2 model
quantized_response_model = torch.quantization.quantize_dynamic(
    response_model, {torch.nn.Linear}, dtype=torch.qint8
)

# Explicitly set the pad_token_id for GPT-2
response_tokenizer.pad_token_id = response_tokenizer.eos_token_id

def generate_response(prompt, max_new_tokens=200):
    try:
        logging.info("Generate response function called")
        inputs = response_tokenizer.encode(prompt, return_tensors='pt')
        logging.info(f"Input length: {len(inputs[0])}")
        attention_mask = inputs.ne(response_tokenizer.pad_token_id).long()
        logging.info(f"Attention mask length: {len(attention_mask[0])}")

        # Try using the quantized model first
        try:
            logging.info("Attempting to use quantized model")
            outputs = quantized_response_model.generate(
                inputs, 
                max_new_tokens=max_new_tokens, 
                num_beams=5, 
                no_repeat_ngram_size=2, 
                early_stopping=True, 
                attention_mask=attention_mask,
                pad_token_id=response_tokenizer.pad_token_id
            )
            logging.info("Quantized model used successfully")
        except Exception as e:
            logging.error(f"Error in quantized model generation: {e}")
            # Fallback to non-quantized model
            logging.info("Falling back to non-quantized model")
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
        return complete_response
    except Exception as e:
        logging.error(f"Error in generate_response: {e}")
        return "An error occurred while generating the response."

# Example usage
prompt = "What does a project manager need?"
response = generate_response(prompt)
print(response)
