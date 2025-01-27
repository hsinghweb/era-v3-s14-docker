from flask import Flask, request, jsonify
import torch
import os

app = Flask(__name__)

# Load the model with error handling
def load_model():
    model_path = os.environ.get('MODEL_PATH', '/app/models/model.pt')
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/generate', methods=['POST'])
def generate_text():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.json
    input_text = data.get('text', '')
    
    # Generate text using the model
    with torch.no_grad():
        # Add your model-specific generation code here
        generated_text = "Generated text based on: " + input_text
    
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 