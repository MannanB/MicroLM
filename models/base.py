import torch
import torch.nn as nn
import torch.nn.functional as F

# from flash_attn import flash_attn_func

import math

def get_sinusoid_encoding_table(max_seq_len, embed_dim, device):
    pe = torch.zeros(max_seq_len, embed_dim, device=device)
    position = torch.arange(0, max_seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # Shape: (max_seq_len, embed_dim)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, use_flash_attn=False, flash_attn_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash_attn = use_flash_attn
        self.flash_attn_dropout = flash_attn_dropout
        self._flash_attn = None
        if self.use_flash_attn:
            try:
                from flash_attn import flash_attn_func
            except ImportError as exc:
                raise ImportError(
                    "flash_attn is enabled in the config but the package is not installed."
                ) from exc
            self._flash_attn = flash_attn_func

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Compute Q, K, V and reshape for multi-head attention.
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_flash_attn:
            attn_output = self._flash_attn(
                q, k, v, dropout_p=self.flash_attn_dropout, causal=True
            )
        else:
            # Scaled dot-product attention.
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)

            if attention_mask is None:
                mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            else:
                if attention_mask.dim() == 2:
                    mask = attention_mask.unsqueeze(0).unsqueeze(0)
                elif attention_mask.dim() == 3:
                    mask = attention_mask.unsqueeze(1)
                elif attention_mask.dim() == 4:
                    mask = attention_mask
                else:
                    raise ValueError(f"Unsupported attention_mask dim {attention_mask.dim()}")
                mask = mask.to(device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)

        # attn_output = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)  # (B, num_heads, T, head_dim)

        # Concatenate heads.
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)
    
class MLP(nn.Module):
    """
    Implements a feed-forward network using two linear layers and GELU activation.
    """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, use_flash_attn=False, flash_attn_dropout=0.0):
        super().__init__()
        self.attn = SelfAttention(
            embed_dim,
            num_heads,
            use_flash_attn=use_flash_attn,
            flash_attn_dropout=flash_attn_dropout,
        )
        self.ff = MLP(embed_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ff(self.ln2(x))

        x = self.dropout(x)

        return x
    

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_seq_len,
        use_flash_attn=False,
        flash_attn_dropout=0.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

        pe = get_sinusoid_encoding_table(max_seq_len, embed_dim, device=torch.device("cpu"))
        self.register_buffer('pe', pe)  # Shape: (max_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                hidden_dim,
                use_flash_attn=use_flash_attn,
                flash_attn_dropout=flash_attn_dropout,
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        token_embeddings = self.token_emb(idx)  # (B, T, embed_dim)

        # Use positional embeddings
        pos_embeddings = self.pe[:T, :].unsqueeze(0)  # (1, T, embed_dim)
        x = token_embeddings + pos_embeddings

        for block in self.layers:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
