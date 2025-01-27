class ModelConfig:
    def __init__(self):
        # LLaMA model configuration
        self.vocab_size = 32000
        self.hidden_size = 256
        self.num_hidden_layers = 12
        self.num_attention_heads = 4  # Changed to match head_dim=64
        self.intermediate_size = 512
        self.hidden_act = "silu"
        self.rms_norm_eps = 1e-5
        self.max_position_embeddings = 1024 