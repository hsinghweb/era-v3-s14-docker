import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # Reshape position_ids to match q's shape
    position_ids = position_ids.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
    
    # Get the rotary embeddings for this position
    cos = cos.squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(0)  # [seq_len, dim]
    
    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim//4, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, seq_len):
        if self.cos_cached is not None and self.cos_cached.size(1) >= seq_len:
            return self.cos_cached[:, :seq_len, :], self.sin_cached[:, :seq_len, :]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = torch.cos(emb)[None, :, :]
        sin = torch.sin(emb)[None, :, :]
        
        self.cos_cached = cos
        self.sin_cached = sin
        
        return cos, sin

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = 64  # Fixed head dimension to match saved model
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, 64, bias=False)  # Single head dimension
        self.v_proj = nn.Linear(config.hidden_size, 64, bias=False)  # Single head dimension
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, rotary_emb=None):
        bsz, q_len, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Split q into heads before applying rotary embeddings
        q = q.view(bsz, q_len, self.num_heads, -1)  # -1 will be 64
        k = k.view(bsz, q_len, 1, -1)  # Keep k as single head
        v = v.view(bsz, q_len, 1, -1)  # Keep v as single head
        
        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            position_ids = torch.arange(q_len, device=q.device)
            cos, sin = rotary_emb(v, q_len)
            # Split q and k in half for rotation
            q1, q2 = q[..., :32], q[..., 32:]
            k1, k2 = k[..., :32], k[..., 32:]
            # Apply rotation to first half
            q_embed = torch.cat([
                q1 * cos.unsqueeze(2) - q2 * sin.unsqueeze(2),
                q2 * cos.unsqueeze(2) + q1 * sin.unsqueeze(2)
            ], dim=-1)
            k_embed = torch.cat([
                k1 * cos.unsqueeze(2) - k2 * sin.unsqueeze(2),
                k2 * cos.unsqueeze(2) + k1 * sin.unsqueeze(2)
            ], dim=-1)
            q, k = q_embed, k_embed
        
        # Expand k and v to match number of heads
        k = k.expand(-1, -1, self.num_heads, -1)
        v = v.expand(-1, -1, self.num_heads, -1)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states, rotary_emb=None):  # Add rotary_emb parameter
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_emb=rotary_emb)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding(dim=64)  # This will create inv_freq of size 16
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_emb=self.rotary_emb)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits, None

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model components
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout
            ),
            num_layers=config.num_layers
        )
        self.fc_out = nn.Linear(config.embedding_dim, config.vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Transformer expects: (seq_len, batch_size, embedding_dim)
        embeddings = embeddings.permute(1, 0, 2)
        
        # Generate attention mask if needed
        mask = self.generate_square_subsequent_mask(embeddings.size(0))
        mask = mask.to(embeddings.device)
        
        # Pass through transformer
        transformer_out = self.transformer(embeddings, mask)
        
        # Back to (batch_size, seq_len, embedding_dim)
        transformer_out = transformer_out.permute(1, 0, 2)
        
        # Project to vocabulary size
        output = self.fc_out(transformer_out)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate(self, input_text, max_length=100):
        # Convert input text to tensor (implement your tokenization here)
        # This is a placeholder - implement actual tokenization
        input_tensor = torch.tensor([[0]])  # Example tensor
        
        generated = []
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                output = self(input_tensor)
                
                # Get next token
                next_token = output[:, -1, :].argmax(dim=-1)
                
                # Append to generated sequence
                generated.append(next_token.item())
                
                # Update input tensor
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                
                # Check for end of sequence token
                if next_token.item() == self.config.eos_token_id:
                    break
        
        # Convert generated tokens to text (implement your detokenization here)
        return f"Generated text for: {input_text}" 