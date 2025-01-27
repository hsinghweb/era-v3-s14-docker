import torch
from torch import nn
from src.models.gpt import LlamaForCausalLM
from src.config.model_config import ModelConfig

class TextGenerator:
    def __init__(self):
        self.model = None
        self.config = None
    
    def generate(self, input_text, max_length=100):
        """
        Generate text using the loaded model
        """
        try:
            # Convert input text to list of token IDs (placeholder - implement proper tokenization)
            # For now, using a simple conversion for demonstration
            input_ids = torch.tensor([[ord(c) % self.config.vocab_size for c in input_text]])
            input_ids = input_ids.to(next(self.model.parameters()).device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = []
                current_input = input_ids
                
                for _ in range(max_length):
                    # Get model predictions
                    outputs, _ = self.model(current_input)
                    next_token_logits = outputs[:, -1, :]
                    
                    # Apply temperature sampling
                    probs = torch.nn.functional.softmax(next_token_logits / 0.7, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated sequence
                    generated_ids.append(next_token.item())
                    
                    # Update input for next iteration
                    current_input = torch.cat([current_input, next_token], dim=1)
                    
                    # Stop if we predict an EOS token
                    if next_token.item() == self.config.vocab_size - 1:  # Assuming last token is EOS
                        break
            
            # Convert generated tokens back to text (placeholder - implement proper detokenization)
            generated_text = input_text + ' ' + ''.join([chr(token % 127) for token in generated_ids])
            return generated_text
            
        except Exception as e:
            print(f"Generation error: {e}")
            raise

def load_model(model_path):
    """
    Load the PyTorch model from the given path
    """
    try:
        generator = TextGenerator()
        
        # Initialize model configuration
        config = ModelConfig()
        generator.config = config
        
        # Create model instance with config
        model = LlamaForCausalLM(config)
        generator.model = model
        
        # Load the checkpoint
        checkpoint = torch.load(
            model_path, 
            map_location=torch.device('cpu'),
            weights_only=True
        )
        
        # Load state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Create new state dict with correct key mapping
            new_state_dict = {}
            for key, value in state_dict.items():
                # Skip cached values
                if 'cached' in key:
                    continue
                    
                if key.startswith('model.'):
                    new_key = key  # Keep as is if already has 'model.' prefix
                else:
                    # Map the keys to the expected format
                    if key == 'embed_tokens.weight':
                        new_key = 'model.embed_tokens.weight'
                    elif key == 'norm.weight':
                        new_key = 'model.norm.weight'
                    elif key == 'rotary_emb.inv_freq':
                        new_key = 'model.rotary_emb.inv_freq'
                    elif key.startswith('layers.'):
                        # Convert layers.0.xxx to model.layers.0.xxx
                        new_key = f"model.{key}"
                    else:
                        new_key = key
                        
                new_state_dict[new_key] = value
                
            # Load the state dict with strict=False to ignore missing keys
            generator.model.load_state_dict(new_state_dict, strict=False)
        
        generator.model.eval()
        return generator
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise 