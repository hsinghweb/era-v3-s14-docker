from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Configuration for ~124M parameters
    block_size: int = 512        # Reduced context length
    vocab_size: int = 50257      # GPT-2 vocabulary size
    n_layer: int = 12           # 12 transformer layers
    n_head: int = 12            # 12 attention heads
    n_embd: int = 768          # 768 embedding dimension
    dropout: float = 0.1        # Dropout for regularization
    bias: bool = True           # Enable bias terms 