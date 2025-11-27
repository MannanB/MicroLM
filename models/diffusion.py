import math
import torch
import torch.nn as nn
from models.base import TransformerBlock, get_sinusoid_encoding_table


def sinusoidal_time_embedding(timesteps, dim):
    """
    timesteps: (B,)
    returns: (B, dim)
    """
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half_dim, device=timesteps.device) / half_dim)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DiffusionRefiner(nn.Module):
    """
    Transformer that denoises interleaved sequences: tokens remain clean and fixed,
    start slots are noised and predicted (epsilon prediction).
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_seq_len,
        num_timesteps=10,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps

        betas = torch.linspace(beta_start, beta_end, steps=num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.start_embed = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.start_embed, mean=0.0, std=1.0 / math.sqrt(embed_dim))

        pe = get_sinusoid_encoding_table(max_seq_len * 2, embed_dim, device=torch.device("cpu"))
        self.register_buffer("pe", pe)

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)  # Predict noise for start slots

    def _get_scheduled_factors(self, timesteps):
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return sqrt_alphas_cumprod, sqrt_one_minus

    def _build_attention_mask(self, token_count, device):
        seq_len = token_count * 2
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

        positions = torch.arange(seq_len, device=device)
        token_positions = positions % 2 == 0
        start_positions = ~token_positions

        token_idx = positions[token_positions]
        start_idx = positions[start_positions]

        if token_idx.numel() > 0 and start_idx.numel() > 0:
            mask[token_idx.unsqueeze(1), start_idx.unsqueeze(0)] = False

        if start_idx.numel() > 0:
            eye = torch.eye(start_idx.numel(), dtype=torch.bool, device=device)
            mask[start_idx.unsqueeze(1), start_idx.unsqueeze(0)] = eye

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, idx, timesteps, noise=None):
        """
        idx: (B, T) token ids
        timesteps: (B,) integer timesteps
        noise: optional noise tensor for start slots (B, T, C). If None, sampled internally.
        returns predicted noise tensor for start slots (B, T, C)
        """
        B, T = idx.shape
        if timesteps.dim() != 1 or timesteps.size(0) != B:
            raise ValueError("timesteps must be shape (B,)")

        token_embeddings = self.token_emb(idx)  # (B, T, C)

        seq_len = T * 2
        if seq_len > self.pe.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds positional embedding limit {self.pe.size(0)}. "
                "Increase max_sequence_length in the config."
            )
        pos_embeddings = self.pe[:seq_len, :].unsqueeze(0)
        token_pos = pos_embeddings[:, ::2, :]
        start_pos = pos_embeddings[:, 1::2, :]

        token_fixed = token_embeddings + token_pos

        start_base = self.start_embed.view(1, 1, self.embed_dim).expand(B, T, self.embed_dim)
        start_base = start_base + start_pos

        if noise is None:
            noise = torch.randn_like(start_base)

        sqrt_alpha, sqrt_one_minus = self._get_scheduled_factors(timesteps)
        x_start_noisy = sqrt_alpha * start_base + sqrt_one_minus * noise

        # Interleave tokens (clean) and noised starts
        stacked = torch.stack((token_fixed, x_start_noisy), dim=2)
        x = stacked.view(B, seq_len, self.embed_dim)

        # Add time embedding to start slots to inform denoising
        t_emb = sinusoidal_time_embedding(timesteps, self.embed_dim)
        t_emb = self.time_mlp(t_emb).view(B, 1, self.embed_dim)
        x[:, 1::2, :] = x[:, 1::2, :] + t_emb

        attn_mask = self._build_attention_mask(T, device=idx.device)

        for block in self.layers:
            x = block(x, attention_mask=attn_mask)

        x = self.ln_f(x)
        pred = self.head(x)
        pred_start_noise = pred[:, 1::2, :]
        return pred_start_noise
