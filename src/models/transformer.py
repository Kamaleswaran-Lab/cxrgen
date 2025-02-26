import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Scaled dot-product attention scaling factor
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Linear projection and reshape for multi-head attention
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute attention output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out)

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Scaled dot-product attention scaling factor
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Linear projection and reshape for multi-head attention
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(encoder_out).chunk(2, dim=-1)
        k, v = map(lambda t: t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2), kv)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute attention output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.pe[:, :x.size(1), :]

class EncoderBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # MLP block
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),  # GELU typically performs better than ReLU
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Stochastic depth (survival probability)
        self.drop_path = nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (typically better than post-norm)
        x1 = self.norm1(x)
        x = x + self.drop_path(self.attention(x1, mask))
        
        x2 = self.norm2(x)
        x = x + self.drop_path(self.mlp(x2))
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention with encoder outputs
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP block
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.drop_path = nn.Identity()
    
    def forward(self, 
                x: torch.Tensor, 
                encoder_out: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x1 = self.norm1(x)
        x = x + self.drop_path(self.self_attn(x1, self_mask))
        # Cross-attention
        x2 = self.norm2(x)
        x = x + self.drop_path(self.cross_attn(x2, encoder_out, cross_mask))
        # MLP
        x3 = self.norm3(x)
        x = x + self.drop_path(self.mlp(x3))
        return x

class EncoderDecoderTransformer(nn.Module):
    def __init__(self,
                 ehr_dim: int = 100,
                 cxr_dim: int = 512,
                 d_model: int = 512,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 max_seq_length: int = 500):
        super().__init__()
        
        # Input embeddings
        self.ehr_embed = nn.Linear(ehr_dim, d_model)
        self.cxr_condition = nn.Linear(cxr_dim, d_model)
        
        # Positional embedding
        self.pos_embed = PositionalEmbedding(d_model, max_seq_length)
        self.pos_drop = nn.Dropout(dropout)
        
        # Encoder layers
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, cxr_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)
    
    def encode(self, 
              ehr: torch.Tensor,
              prev_cxr: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode EHR data using the transformer encoder."""
        B, T, _ = ehr.shape
        
        # Project inputs to d_model dimension
        x = self.ehr_embed(ehr)
        
        # Add CXR condition to each timestep
        cxr_cond = self.cxr_condition(prev_cxr)
        x = x + cxr_cond
        
        # Add positional embedding
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        
        # Create attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        else:
            mask = None
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x, mask)
        
        # Final encoder norm
        x = self.encoder_norm(x)
        return x
    
    def decode(self, 
              x: torch.Tensor,
              encoder_out: torch.Tensor,
              self_attention_mask: Optional[torch.Tensor] = None,
              cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode using transformer decoder with encoder-decoder attention."""
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x, encoder_out, self_attention_mask, cross_attention_mask)
            print(x[0,:, :])
        # Final decoder norm
        
        x = self.decoder_norm(x)
        return x
    
    def forward(self, 
                ehr: torch.Tensor,
                prev_cxr: torch.Tensor,
                target_input: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                causal_mask: bool = True) -> torch.Tensor:
        """
        Forward pass through the entire encoder-decoder transformer.
        
        Args:
            ehr: Tensor of shape [batch_size, seq_length, ehr_dim]
            prev_cxr: Tensor of shape [batch_size, cxr_dim]
            target_input: Optional tensor for teacher forcing, shape [batch_size, seq_length, cxr_dim]
                          If None, uses zero tensor initialized to proper size.
            encoder_attention_mask: Optional mask for encoder [batch_size, seq_length]
            decoder_attention_mask: Optional mask for decoder [batch_size, seq_length]
            causal_mask: Whether to apply causal masking in the decoder
        """
        B, T, _ = ehr.shape
        
        # Run encoder
        encoder_out = self.encode(ehr, prev_cxr, encoder_attention_mask)
        
        # Initialize decoder input (or use teacher forcing input)
        if target_input is None:
            # Start with zeros or learned query tokens
            decoder_input = torch.zeros(B, T, self.head.out_features, device=ehr.device)
        else:
            decoder_input = target_input
        
        # Project to d_model
        decoder_input = nn.Linear(self.head.out_features, self.encoder_blocks[0].attention.d_model).to(ehr.device)(decoder_input)

        decoder_input = self.pos_embed(decoder_input)

        # Create causal mask if needed
        if causal_mask:
            causal_mask_tensor = torch.triu(
                torch.ones(T, T, device=ehr.device) * float('-inf'), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            
            # Combine with provided mask if any
            if decoder_attention_mask is not None:
                mask = decoder_attention_mask.unsqueeze(1).unsqueeze(2)
                causal_mask_tensor = causal_mask_tensor.masked_fill(mask == 0, float('-inf'))
        else:
            causal_mask_tensor = None if decoder_attention_mask is None else decoder_attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Run decoder
        decoder_out = self.decode(
            decoder_input, 
            encoder_out,
            causal_mask_tensor,
            encoder_attention_mask.unsqueeze(1).unsqueeze(2) if encoder_attention_mask is not None else None
        )
        
        # Project to output dimension
        out = self.head(decoder_out)
        
        return out

def create_transformer_model(config: dict) -> EncoderDecoderTransformer:
    """
    Factory function to create transformer model with specified configuration.
    
    Example config:
    {
        'ehr_dim': 100,
        'cxr_dim': 512,
        'd_model': 512,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'max_seq_length': 500
    }
    """
    return EncoderDecoderTransformer(**config)


# Example usage
if __name__ == "__main__":
    # Model configuration
    config = {
        'ehr_dim': 100,
        'cxr_dim': 512,
        'd_model': 512,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'max_seq_length': 500
    }
    
    # Create model
    model = create_transformer_model(config)
    
    # Example forward pass
    batch_size = 8
    seq_length = 100
    
    ehr = torch.randn(batch_size, seq_length, config['ehr_dim'])
    prev_cxr = torch.randn(batch_size, config['cxr_dim'])
    mask = torch.ones(batch_size, seq_length)
    
    # Get predictions
    predictions = model(ehr, prev_cxr, None, mask, mask)
    print(f"Output shape: {predictions.shape}")  # [batch_size, seq_length, cxr_dim]