import requests
import time
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s [CLIENT] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def send_request(text, max_tokens=100):
    url = 'http://server:5000/generate'
    data = {
        'text': text,
        'max_tokens': max_tokens
    }
    
    logging.info(f"Sending request - Text: '{text}', Max tokens: {max_tokens}")
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise exception for bad status codes
        result = response.json()
        
        if 'error' in result:
            logging.error(f"Server returned error: {result['error']}")
            return None
            
        generated_text = result['generated_text']
        logging.info(f"Received response: '{generated_text}'")
        return generated_text
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        return None

if __name__ == '__main__':
    # Wait for the server to start
    logging.info("Waiting for server to start...")
    time.sleep(5)
    logging.info("Client ready")
    
    while True:
        try:
            user_input = input("\nEnter text to generate (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                logging.info("Shutting down client")
                break
            
            max_tokens = input("Enter number of tokens to generate (default 100): ")
            max_tokens = int(max_tokens) if max_tokens.isdigit() else 100
            
            generated_text = send_request(user_input, max_tokens)
            if generated_text:
                print("\nGenerated text:")
                print("-" * 50)
                print(generated_text)
                print("-" * 50)
            
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}") 