import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

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
        
        # Use PyTorch's MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # MLP block
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, debug = False, batch_idx_debug = 0) -> torch.Tensor:
        # Pre-norm architecture
        x_norm = self.norm1(x)
        
        # Handle mask format for PyTorch's MultiheadAttention
        key_padding_mask = None
        attn_mask = None
        
        if mask is not None:
            # For proper causal masking in a square matrix [T, T]
            if mask.dim() == 2 and mask.size(0) == mask.size(1):
                attn_mask = mask
            # If it's a 4D mask [B, 1, 1, T], convert to key_padding_mask [B, T]
            elif mask.dim() == 4:
                key_padding_mask = ~mask.squeeze(1).squeeze(1).bool()
            # If it's a 2D mask [B, T], use as key_padding_mask
            elif mask.dim() == 2:
                # Ensure the mask has shape [B, T] not [T, B]
                if mask.size(0) != x.size(0):
                    if mask.size(0) == x.size(1) and mask.size(1) == x.size(0):
                        # Transpose if needed
                        mask = mask.transpose(0, 1)
                key_padding_mask = ~mask.bool()

        if debug:
            print("key padding mask")
            print(key_padding_mask[batch_idx_debug, :])
            print("Attention mask")
            print(attn_mask)
            print("x_norm")
            print(x_norm[batch_idx_debug, :, :])
            print(f"x_norm - min: {x_norm.min().item()}, max: {x_norm.max().item()}, "
            f"has_nan: {torch.isnan(x_norm).any().item()}")
    
        # Self-attention with residual connection
        attn_out, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        if debug:
            print("Attention output")
            print(attn_out[batch_idx_debug, :, :])

        x = x + self.dropout(attn_out)
        
        # MLP with residual connection
        x_norm = self.norm2(x)
        x = x + self.dropout(self.mlp(x_norm))
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
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
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                encoder_out: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm for self-attention
        x_norm = self.norm1(x)
        
        # Handle self-attention mask format for PyTorch's MultiheadAttention
        self_key_padding_mask = None
        self_attn_mask = None
        
        if self_mask is not None:
            # For causal mask which should be a square matrix
            if isinstance(self_mask, torch.Tensor) and self_mask.dim() == 2 and self_mask.size(0) == self_mask.size(1):
                self_attn_mask = self_mask  # Use directly as attn_mask
            # If it's a 4D mask [B, 1, 1, T], convert to key_padding_mask [B, T]
            elif self_mask.dim() == 4:
                self_key_padding_mask = ~self_mask.squeeze(1).squeeze(1).bool()
            # If it's a 2D mask [B, T], ensure it's in the right format
            elif self_mask.dim() == 2:
                if self_mask.size(0) != x.size(0):
                    # Transpose if dimensions are swapped
                    if self_mask.size(0) == x.size(1) and self_mask.size(1) == x.size(0):
                        self_mask = self_mask.transpose(0, 1)
                self_key_padding_mask = ~self_mask.bool()
            
        # Self-attention
        self_attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=self_key_padding_mask,
            attn_mask=self_attn_mask,
            need_weights=False
        )
        x = x + self.dropout(self_attn_out)
        
        # Pre-norm for cross-attention
        x_norm = self.norm2(x)
        
        # Handle cross-attention mask format
        cross_key_padding_mask = None
        cross_attn_mask = None
        
        if cross_mask is not None:
            # If it's a 4D mask [B, 1, 1, T], convert to key_padding_mask [B, T]
            if cross_mask.dim() == 4:
                cross_key_padding_mask = ~cross_mask.squeeze(1).squeeze(1).bool()
            # If it's a 2D mask [B, T], ensure it's in the right format
            elif cross_mask.dim() == 2:
                if cross_mask.size(0) != x.size(0):
                    # Transpose if dimensions are swapped
                    if cross_mask.size(0) == encoder_out.size(1) and cross_mask.size(1) == x.size(0):
                        cross_mask = cross_mask.transpose(0, 1)
                cross_key_padding_mask = ~cross_mask.bool()
            
        # Cross-attention
        cross_attn_out, _ = self.cross_attn(
            query=x_norm,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=cross_key_padding_mask,
            attn_mask=cross_attn_mask,
            need_weights=False
        )
        x = x + self.dropout(cross_attn_out)
        
        # Pre-norm for MLP
        x_norm = self.norm3(x)
        x = x + self.dropout(self.mlp(x_norm))
        
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
        
        # Encoder layers - could use TransformerEncoder but using individual blocks for architecture consistency
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers - could use TransformerDecoder but using individual blocks for architecture consistency
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
              attention_mask: Optional[torch.Tensor] = None, debug = False, batch_idx_debug = 0) -> torch.Tensor:
        """Encode EHR data using the transformer encoder."""
        B, T, _ = ehr.shape
        
        if debug:
            print("EHR input")
            print(ehr[batch_idx_debug, :, :])
            print(f"ehr - min: {ehr.min().item()}, max: {ehr.max().item()}, has_nan: {torch.isnan(ehr).any().item()}")

        # Project inputs to d_model dimension
        x = self.ehr_embed(ehr)
        
        if debug:
            print("EHR embedding")
            print(x[batch_idx_debug, :, :])
            print(f"x - min: {x.min().item()}, max: {x.max().item()}, has_nan: {torch.isnan(x).any().item()}")

        # Add CXR condition to each timestep
        cxr_cond = self.cxr_condition(prev_cxr)

        if debug:
            print("CXR condition")
            print(cxr_cond[batch_idx_debug, :, :])
            print(f"cxr_cond - min: {cxr_cond.min().item()}, max: {cxr_cond.max().item()}, has_nan: {torch.isnan(cxr_cond).any().item()}")

        x = x + cxr_cond

        if debug:
            print("EHR embedding with CXR condition")
            print(x[batch_idx_debug, :, :])
            print(f"x - min: {x.min().item()}, max: {x.max().item()}, has_nan: {torch.isnan(x).any().item()}")
        
        # Add positional embedding
        x = self.pos_embed(x)

        if debug:
            print("EHR embedding with CXR condition and positional embedding")
            print(x[batch_idx_debug, :, :])
            print(f"x - min: {x.min().item()}, max: {x.max().item()}, has_nan: {torch.isnan(x).any().item()}")

        x = self.pos_drop(x)

        if debug:
            print("EHR embedding with CXR condition and positional embedding after dropout")
            print(x[batch_idx_debug, :, :])
            print(f"x - min: {x.min().item()}, max: {x.max().item()}, has_nan: {torch.isnan(x).any().item()}")
        
        # Create attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        else:
            mask = None
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x, mask, debug = debug, batch_idx_debug = batch_idx_debug)
        
        if debug:
            print("Encoder output")
            print(x[batch_idx_debug, :, :])
        
        
        # Final encoder norm
        x = self.encoder_norm(x)

        if debug:
            print("Encoder output after norm")
            print(x[batch_idx_debug, :, :])
        return x
    
    def decode(self, 
              x: torch.Tensor,
              encoder_out: torch.Tensor,
              self_attention_mask: Optional[torch.Tensor] = None,
              cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode using transformer decoder with encoder-decoder attention."""
        # Apply decoder blocks
        for block in self.decoder_blocks:
            # Pass the masks directly to the decoder block
            x = block(
                x=x,
                encoder_out=encoder_out,
                self_mask=self_attention_mask,
                cross_mask=cross_attention_mask
            )
        
        # Final decoder norm
        x = self.decoder_norm(x)
        return x
    
    def forward(self, 
                ehr: torch.Tensor,
                prev_cxr: torch.Tensor,
                target_input: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                causal_mask: bool = True, debug = False, batch_idx_debug = 0) -> torch.Tensor:
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
        encoder_out = self.encode(ehr, prev_cxr, encoder_attention_mask, debug=debug, batch_idx_debug=batch_idx_debug)
        
        if debug:
            print("Encoder output shape:")
            print(encoder_out[batch_idx_debug, :, :])

        # Initialize decoder input (or use teacher forcing input)
        target_input = None
        if target_input is None:
            # Start with zeros or learned query tokens
            decoder_input = torch.zeros(B, T, self.head.out_features, device=ehr.device)
        else:
            decoder_input = target_input
        
        # Project to d_model
        decoder_input = nn.Linear(self.head.out_features, self.encoder_blocks[0].attention.embed_dim).to(ehr.device)(decoder_input)
        
        if debug:
            print("Decoder input")
            print(decoder_input[batch_idx_debug, :, :])

        # Add positional embeddings
        decoder_input = self.pos_embed(decoder_input)
        
        if debug:
            print("Decoder input after positional embeddings")
            print(decoder_input[batch_idx_debug, :, :])

        # Create causal mask for decoder self-attention if needed
        causal_attn_mask = None
        if causal_mask:
            # Create a causal mask as expected by PyTorch's MultiheadAttention: 
            # a 2D square matrix of shape [seq_len, seq_len]
            seq_len = decoder_input.size(1)
            causal_attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=ehr.device) * float('-inf'), 
                diagonal=1
            
            )  # Shape: [seq_len, seq_len]
            if debug:
                print("Causal mask")
                print(causal_attn_mask)

        # Handle decoder padding mask
        decoder_key_padding_mask = None
        if decoder_attention_mask is not None:
            # Make sure padding mask is in the right format [batch_size, seq_len]
            if decoder_attention_mask.size(0) != B:
                # If dimensions are swapped, transpose
                if decoder_attention_mask.size(0) == T and decoder_attention_mask.size(1) == B:
                    decoder_attention_mask = decoder_attention_mask.transpose(0, 1)
            # Convert to format expected by PyTorch (True = position to mask)
            decoder_key_padding_mask = ~decoder_attention_mask.bool()
        
        # Handle encoder padding mask for cross-attention
        encoder_key_padding_mask = None
        if encoder_attention_mask is not None:
            # Make sure mask has shape [batch_size, encoder_seq_len]
            if encoder_attention_mask.size(0) != B:
                # If dimensions are swapped, transpose
                if encoder_attention_mask.size(0) == T and encoder_attention_mask.size(1) == B:
                    encoder_attention_mask = encoder_attention_mask.transpose(0, 1)
            # Convert to format expected by PyTorch (True = position to mask)
            encoder_key_padding_mask = ~encoder_attention_mask.bool()
        
        # Apply decoder layers
        x = decoder_input
        for block in self.decoder_blocks:
            # Pre-norm for self-attention
            x_norm = block.norm1(x)
            
            if debug:
                print("Decoder block input")
                print(x_norm[batch_idx_debug, :, :])

            # Self-attention with causal and padding masks
            self_attn_out, _ = block.self_attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                key_padding_mask=decoder_key_padding_mask,
                attn_mask=causal_attn_mask,
                need_weights=False
            )
            if debug:
                print("Decoder block self-attention output")
                print(self_attn_out[batch_idx_debug, :, :])

            x = x + block.dropout(self_attn_out)
            
            if debug:
                print("Decoder block self-attention output")
                print(x[batch_idx_debug, :, :])
            # Pre-norm for cross-attention
            x_norm = block.norm2(x)
            
            if debug:
                print("Decoder block cross-attention input")
                print(x_norm[batch_idx_debug, :, :])

            # Cross-attention with encoder padding mask
            cross_attn_out, _ = block.cross_attn(
                query=x_norm,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_key_padding_mask,
                attn_mask=None,
                need_weights=False
            )

            if debug:
                print("Decoder block cross-attention output")
                print(cross_attn_out[batch_idx_debug, :, :])

            x = x + block.dropout(cross_attn_out)
            
            if debug:
                print("Decoder block cross-attention output")
                print(x[batch_idx_debug, :, :])

            # MLP
            x_norm = block.norm3(x)

            if debug:
                print("Decoder block MLP input")
                print(x_norm[batch_idx_debug, :, :])

            x = x + block.dropout(block.mlp(x_norm))
            if debug:
                print("Decoder block MLP output")
                print(x[batch_idx_debug, :, :])

        # Final decoder norm
        decoder_out = self.decoder_norm(x)
        
        if debug:
            print("Decoder output")
            print(decoder_out[batch_idx_debug, :, :])

        # Project to output dimension
        out = self.head(decoder_out)

        if debug:
            print("Output")
            print(out[batch_idx_debug, :, :])
        
        return out
    

class EncoderDecoderTransformerConcat(nn.Module):
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
        self.ehr_cxr_fuse = nn.Linear(ehr_dim + cxr_dim, d_model)
        
        # Positional embedding
        self.pos_embed = PositionalEmbedding(d_model, max_seq_length)
        self.pos_drop = nn.Dropout(dropout)
        
        # Encoder layers - could use TransformerEncoder but using individual blocks for architecture consistency
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers - could use TransformerDecoder but using individual blocks for architecture consistency
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
              attention_mask: Optional[torch.Tensor] = None, debug = False, batch_idx_debug = 0) -> torch.Tensor:
        """Encode EHR data using the transformer encoder."""
        B, T, _ = ehr.shape
        
        if debug:
            print("EHR input")
            print(ehr[batch_idx_debug, :, :])
            print(f"ehr - min: {ehr.min().item()}, max: {ehr.max().item()}, has_nan: {torch.isnan(ehr).any().item()}")

        # Project inputs to d_model dimension
        ehr_cxr = torch.cat([ehr, prev_cxr], dim=-1)
        x = self.ehr_cxr_fuse(ehr_cxr)

        
        if debug:
            print("EHR embedding")
            print(x[batch_idx_debug, :, :])
            print(f"x - min: {x.min().item()}, max: {x.max().item()}, has_nan: {torch.isnan(x).any().item()}")

        # Add positional embedding
        x = self.pos_embed(x)

        x = self.pos_drop(x)

        if debug:
            print("EHR embedding with CXR condition and positional embedding after dropout")
            print(x[batch_idx_debug, :, :])
            print(f"x - min: {x.min().item()}, max: {x.max().item()}, has_nan: {torch.isnan(x).any().item()}")
        
        # Create attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        else:
            mask = None
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x, mask, debug = debug, batch_idx_debug = batch_idx_debug)
        
        if debug:
            print("Encoder output")
            print(x[batch_idx_debug, :, :])
        
        
        # Final encoder norm
        x = self.encoder_norm(x)

        if debug:
            print("Encoder output after norm")
            print(x[batch_idx_debug, :, :])
        return x
    
    def decode(self, 
              x: torch.Tensor,
              encoder_out: torch.Tensor,
              self_attention_mask: Optional[torch.Tensor] = None,
              cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode using transformer decoder with encoder-decoder attention."""
        # Apply decoder blocks
        for block in self.decoder_blocks:
            # Pass the masks directly to the decoder block
            x = block(
                x=x,
                encoder_out=encoder_out,
                self_mask=self_attention_mask,
                cross_mask=cross_attention_mask
            )
        
        # Final decoder norm
        x = self.decoder_norm(x)
        return x
    
    def forward(self, 
                ehr: torch.Tensor,
                prev_cxr: torch.Tensor,
                target_input: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                causal_mask: bool = True, debug = False, batch_idx_debug = 0) -> torch.Tensor:
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
        encoder_out = self.encode(ehr, prev_cxr, encoder_attention_mask, debug=debug, batch_idx_debug=batch_idx_debug)
        
        if debug:
            print("Encoder output shape:")
            print(encoder_out[batch_idx_debug, :, :])

        target_input = None

        # Initialize decoder input (or use teacher forcing input)
        if target_input is None:
            # Start with zeros or learned query tokens
            decoder_input = torch.zeros(B, T, self.head.out_features, device=ehr.device)
        else:
            decoder_input = target_input
        
        # Project to d_model
        decoder_input = nn.Linear(self.head.out_features, self.encoder_blocks[0].attention.embed_dim).to(ehr.device)(decoder_input)
        
        if debug:
            print("Decoder input")
            print(decoder_input[batch_idx_debug, :, :])

        # Add positional embeddings
        decoder_input = self.pos_embed(decoder_input)
        
        if debug:
            print("Decoder input after positional embeddings")
            print(decoder_input[batch_idx_debug, :, :])

        # Create causal mask for decoder self-attention if needed
        causal_attn_mask = None
        if causal_mask:
            # Create a causal mask as expected by PyTorch's MultiheadAttention: 
            # a 2D square matrix of shape [seq_len, seq_len]
            seq_len = decoder_input.size(1)
            causal_attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=ehr.device) * float('-inf'), 
                diagonal=1
            
            )  # Shape: [seq_len, seq_len]
            if debug:
                print("Causal mask")
                print(causal_attn_mask)

        # Handle decoder padding mask
        decoder_key_padding_mask = None
        if decoder_attention_mask is not None:
            # Make sure padding mask is in the right format [batch_size, seq_len]
            if decoder_attention_mask.size(0) != B:
                # If dimensions are swapped, transpose
                if decoder_attention_mask.size(0) == T and decoder_attention_mask.size(1) == B:
                    decoder_attention_mask = decoder_attention_mask.transpose(0, 1)
            # Convert to format expected by PyTorch (True = position to mask)
            decoder_key_padding_mask = ~decoder_attention_mask.bool()
        
        # Handle encoder padding mask for cross-attention
        encoder_key_padding_mask = None
        if encoder_attention_mask is not None:
            # Make sure mask has shape [batch_size, encoder_seq_len]
            if encoder_attention_mask.size(0) != B:
                # If dimensions are swapped, transpose
                if encoder_attention_mask.size(0) == T and encoder_attention_mask.size(1) == B:
                    encoder_attention_mask = encoder_attention_mask.transpose(0, 1)
            # Convert to format expected by PyTorch (True = position to mask)
            encoder_key_padding_mask = ~encoder_attention_mask.bool()
        
        # Apply decoder layers
        x = decoder_input
        for block in self.decoder_blocks:
            # Pre-norm for self-attention
            x_norm = block.norm1(x)
            
            if debug:
                print("Decoder block input")
                print(x_norm[batch_idx_debug, :, :])

            # Self-attention with causal and padding masks
            self_attn_out, _ = block.self_attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                key_padding_mask=decoder_key_padding_mask,
                attn_mask=causal_attn_mask,
                need_weights=False
            )
            if debug:
                print("Decoder block self-attention output")
                print(self_attn_out[batch_idx_debug, :, :])

            x = x + block.dropout(self_attn_out)
            
            if debug:
                print("Decoder block self-attention output")
                print(x[batch_idx_debug, :, :])
            # Pre-norm for cross-attention
            x_norm = block.norm2(x)
            
            if debug:
                print("Decoder block cross-attention input")
                print(x_norm[batch_idx_debug, :, :])

            # Cross-attention with encoder padding mask
            cross_attn_out, _ = block.cross_attn(
                query=x_norm,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_key_padding_mask,
                attn_mask=None,
                need_weights=False
            )

            if debug:
                print("Decoder block cross-attention output")
                print(cross_attn_out[batch_idx_debug, :, :])

            x = x + block.dropout(cross_attn_out)
            
            if debug:
                print("Decoder block cross-attention output")
                print(x[batch_idx_debug, :, :])

            # MLP
            x_norm = block.norm3(x)

            if debug:
                print("Decoder block MLP input")
                print(x_norm[batch_idx_debug, :, :])

            x = x + block.dropout(block.mlp(x_norm))
            if debug:
                print("Decoder block MLP output")
                print(x[batch_idx_debug, :, :])

        # Final decoder norm
        decoder_out = self.decoder_norm(x)
        
        if debug:
            print("Decoder output")
            print(decoder_out[batch_idx_debug, :, :])

        # Project to output dimension
        out = self.head(decoder_out)

        if debug:
            print("Output")
            print(out[batch_idx_debug, :, :])
        
        return out
    
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256]):
        super(MLPModel, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.BatchNorm1d(size)) 
            prev_size = size
            
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)  

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten (batch_size, 512, 1) â†’ (batch_size, 512)

        for i in range(0, len(self.layers), 2):  
            x = self.layers[i](x)  # Linear layer
            x = torch.relu(x)  
            x = self.layers[i + 1](x)  # BatchNorm layer

        final_hidden = x  
        output = self.output_layer(final_hidden)  

        return output, final_hidden 
    
LABEL_COLUMNS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices'
]
    

class EncoderDecoderTransformerConcatMLP(nn.Module):
    def __init__(self, classifier_path: str,
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
        
        self.classifier_path = classifier_path
        # Input embeddings
        self.ehr_cxr_fuse = nn.Linear(ehr_dim + cxr_dim, d_model)
        
        # Positional embedding
        self.pos_embed = PositionalEmbedding(d_model, max_seq_length)
        self.pos_drop = nn.Dropout(dropout)
        
        # Encoder layers - could use TransformerEncoder but using individual blocks for architecture consistency
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers - could use TransformerDecoder but using individual blocks for architecture consistency
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
        self.mlp = MLPModel(cxr_dim, len(LABEL_COLUMNS))
        self.mlp = torch.load(self.classifier_path, weights_only = False)

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
              attention_mask: Optional[torch.Tensor] = None, debug = False, batch_idx_debug = 0) -> torch.Tensor:
        """Encode EHR data using the transformer encoder."""
        B, T, _ = ehr.shape
        
        if debug:
            print("EHR input")
            print(ehr[batch_idx_debug, :, :])
            print(f"ehr - min: {ehr.min().item()}, max: {ehr.max().item()}, has_nan: {torch.isnan(ehr).any().item()}")

        # Project inputs to d_model dimension
        ehr_cxr = torch.cat([ehr, prev_cxr], dim=-1)
        x = self.ehr_cxr_fuse(ehr_cxr)

        
        if debug:
            print("EHR embedding")
            print(x[batch_idx_debug, :, :])
            print(f"x - min: {x.min().item()}, max: {x.max().item()}, has_nan: {torch.isnan(x).any().item()}")

        # Add positional embedding
        x = self.pos_embed(x)

        x = self.pos_drop(x)

        if debug:
            print("EHR embedding with CXR condition and positional embedding after dropout")
            print(x[batch_idx_debug, :, :])
            print(f"x - min: {x.min().item()}, max: {x.max().item()}, has_nan: {torch.isnan(x).any().item()}")
        
        # Create attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        else:
            mask = None
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x, mask, debug = debug, batch_idx_debug = batch_idx_debug)
        
        if debug:
            print("Encoder output")
            print(x[batch_idx_debug, :, :])
        
        
        # Final encoder norm
        x = self.encoder_norm(x)

        if debug:
            print("Encoder output after norm")
            print(x[batch_idx_debug, :, :])
        return x
    
    def decode(self, 
              x: torch.Tensor,
              encoder_out: torch.Tensor,
              self_attention_mask: Optional[torch.Tensor] = None,
              cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode using transformer decoder with encoder-decoder attention."""
        # Apply decoder blocks
        for block in self.decoder_blocks:
            # Pass the masks directly to the decoder block
            x = block(
                x=x,
                encoder_out=encoder_out,
                self_mask=self_attention_mask,
                cross_mask=cross_attention_mask
            )
        
        # Final decoder norm
        x = self.decoder_norm(x)
        return x
    
    def forward(self, 
                ehr: torch.Tensor,
                prev_cxr: torch.Tensor,
                target_input: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                causal_mask: bool = True, debug = False, batch_idx_debug = 0) -> torch.Tensor:
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
        encoder_out = self.encode(ehr, prev_cxr, encoder_attention_mask, debug=debug, batch_idx_debug=batch_idx_debug)
        
        if debug:
            print("Encoder output shape:")
            print(encoder_out[batch_idx_debug, :, :])

        target_input = None

        # Initialize decoder input (or use teacher forcing input)
        if target_input is None:
            # Start with zeros or learned query tokens
            decoder_input = torch.zeros(B, T, self.head.out_features, device=ehr.device)
        else:
            decoder_input = target_input
        
        # Project to d_model
        decoder_input = nn.Linear(self.head.out_features, self.encoder_blocks[0].attention.embed_dim).to(ehr.device)(decoder_input)
        
        if debug:
            print("Decoder input")
            print(decoder_input[batch_idx_debug, :, :])

        # Add positional embeddings
        decoder_input = self.pos_embed(decoder_input)
        
        if debug:
            print("Decoder input after positional embeddings")
            print(decoder_input[batch_idx_debug, :, :])

        # Create causal mask for decoder self-attention if needed
        causal_attn_mask = None
        if causal_mask:
            # Create a causal mask as expected by PyTorch's MultiheadAttention: 
            # a 2D square matrix of shape [seq_len, seq_len]
            seq_len = decoder_input.size(1)
            causal_attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=ehr.device) * float('-inf'), 
                diagonal=1
            
            )  # Shape: [seq_len, seq_len]
            if debug:
                print("Causal mask")
                print(causal_attn_mask)

        # Handle decoder padding mask
        decoder_key_padding_mask = None
        if decoder_attention_mask is not None:
            # Make sure padding mask is in the right format [batch_size, seq_len]
            if decoder_attention_mask.size(0) != B:
                # If dimensions are swapped, transpose
                if decoder_attention_mask.size(0) == T and decoder_attention_mask.size(1) == B:
                    decoder_attention_mask = decoder_attention_mask.transpose(0, 1)
            # Convert to format expected by PyTorch (True = position to mask)
            decoder_key_padding_mask = ~decoder_attention_mask.bool()
        
        # Handle encoder padding mask for cross-attention
        encoder_key_padding_mask = None
        if encoder_attention_mask is not None:
            # Make sure mask has shape [batch_size, encoder_seq_len]
            if encoder_attention_mask.size(0) != B:
                # If dimensions are swapped, transpose
                if encoder_attention_mask.size(0) == T and encoder_attention_mask.size(1) == B:
                    encoder_attention_mask = encoder_attention_mask.transpose(0, 1)
            # Convert to format expected by PyTorch (True = position to mask)
            encoder_key_padding_mask = ~encoder_attention_mask.bool()
        
        # Apply decoder layers
        x = decoder_input
        for block in self.decoder_blocks:
            # Pre-norm for self-attention
            x_norm = block.norm1(x)
            
            if debug:
                print("Decoder block input")
                print(x_norm[batch_idx_debug, :, :])

            # Self-attention with causal and padding masks
            self_attn_out, _ = block.self_attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                key_padding_mask=decoder_key_padding_mask,
                attn_mask=causal_attn_mask,
                need_weights=False
            )
            if debug:
                print("Decoder block self-attention output")
                print(self_attn_out[batch_idx_debug, :, :])

            x = x + block.dropout(self_attn_out)
            
            if debug:
                print("Decoder block self-attention output")
                print(x[batch_idx_debug, :, :])
            # Pre-norm for cross-attention
            x_norm = block.norm2(x)
            
            if debug:
                print("Decoder block cross-attention input")
                print(x_norm[batch_idx_debug, :, :])

            # Cross-attention with encoder padding mask
            cross_attn_out, _ = block.cross_attn(
                query=x_norm,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_key_padding_mask,
                attn_mask=None,
                need_weights=False
            )

            if debug:
                print("Decoder block cross-attention output")
                print(cross_attn_out[batch_idx_debug, :, :])

            x = x + block.dropout(cross_attn_out)
            
            if debug:
                print("Decoder block cross-attention output")
                print(x[batch_idx_debug, :, :])

            # MLP
            x_norm = block.norm3(x)

            if debug:
                print("Decoder block MLP input")
                print(x_norm[batch_idx_debug, :, :])

            x = x + block.dropout(block.mlp(x_norm))
            if debug:
                print("Decoder block MLP output")
                print(x[batch_idx_debug, :, :])

        # Final decoder norm
        decoder_out = self.decoder_norm(x)
        
        if debug:
            print("Decoder output")
            print(decoder_out[batch_idx_debug, :, :])

        # Project to output dimension
        out = self.head(decoder_out)
        classifier_out, classifier_hidden = self.mlp(out)

        if debug:
            print("Output")
            print(out[batch_idx_debug, :, :])
        
        return out, classifier_out
        

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

def create_transformerconcat_model(config: dict) -> EncoderDecoderTransformer:
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
    return EncoderDecoderTransformerConcat(**config)

def create_transformerconcatclassifier_model(config: dict) -> EncoderDecoderTransformer:
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
    return EncoderDecoderTransformerConcatMLP(**config)