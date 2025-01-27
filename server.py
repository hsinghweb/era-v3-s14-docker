from flask import Flask, request, jsonify
import torch
from src.model_loader import load_model
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s [SERVER] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)

# Load the model
logging.info("Loading model...")
model = load_model("model.pt")
logging.info("Model loaded successfully")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data.get('text', '')
    max_tokens = data.get('max_tokens', 100)
    
    logging.info(f"Received request - Input text: '{input_text}', Max tokens: {max_tokens}")
    
    try:
        # Generate text using the model
        logging.info("Generating text...")
        generated_text = model.generate(input_text, max_length=max_tokens)
        
        response = {'generated_text': generated_text}
        logging.info(f"Generated response: '{generated_text}'")
        
    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        logging.error(error_msg)
        response = {'error': error_msg}
    
    return jsonify(response)

if __name__ == '__main__':
    logging.info("Starting server on port 5000...")
    app.run(host='0.0.0.0', port=5000) 