import torch
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[SERVER] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
try:
    model = torch.load('model.pt')
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('text', '')
    
    logger.info(f"Received request with input: {input_text}")
    
    try:
        # Add your text generation logic here using the model
        # This is a placeholder - replace with your actual model inference
        generated_text = f"Generated response for: {input_text}"
        
        logger.info(f"Generated text: {generated_text}")
        return jsonify({'response': generated_text})
    
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting server on port 5000")
    app.run(host='0.0.0.0', port=5000) 