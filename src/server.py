import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import logging
from config.model_config import GPTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[SERVER] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the same model architecture used during training
class TextGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.lstm = nn.LSTM(config.n_embd, config.n_embd, 
                           num_layers=config.n_layer,
                           dropout=config.dropout if config.n_layer > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

app = Flask(__name__)

# Load the model with CPU mapping
try:
    # Initialize model with config
    config = GPTConfig()
    model = TextGenerator(config)
    
    # Load the state dict
    state_dict = torch.load('model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
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