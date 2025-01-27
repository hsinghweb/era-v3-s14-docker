import requests
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[CLIENT] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_text(text, server_url="http://server:5000/generate"):
    logger.info(f"Sending request to server with input: {text}")
    
    try:
        response = requests.post(
            server_url,
            json={'text': text}
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Received response from server: {result['response']}")
        return result['response']
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with server: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "Default input text"
    
    generated_text = generate_text(input_text)
    if generated_text:
        print("\nGenerated Text:", generated_text) 