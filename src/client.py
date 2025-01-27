import requests
import os
import time

def main():
    server_url = os.environ.get('SERVER_URL', 'http://model-server:5000')
    
    while True:
        try:
            # Get input from user
            input_text = input("Enter text to generate (or 'quit' to exit): ")
            
            if input_text.lower() == 'quit':
                break
            
            # Send request to model server
            response = requests.post(
                f"{server_url}/generate",
                json={'text': input_text}
            )
            
            # Display the response
            if response.status_code == 200:
                result = response.json()
                print("Generated text:", result['generated_text'])
            else:
                print("Error:", response.status_code)
                
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)  # Wait before retrying

if __name__ == '__main__':
    main() 