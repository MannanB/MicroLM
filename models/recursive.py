import math

from models.base import *


class RecursiveTransformer(nn.Module):
    """
    Decoder-only transformer that interleaves a learnable start embedding between
    every token so the model can iteratively refine the start slots. The sequence
    length doubles (token, start, token, start, ...). During training, consumers
    can supervise both the token slots (identity) and the start slots (next-token).
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_seq_len,
        n_recursions=4,
        return_all_logits=False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.n_recursions = n_recursions
        self.max_seq_len = max_seq_len
        self.default_return_all_logits = return_all_logits

        pe = get_sinusoid_encoding_table(max_seq_len * 2, embed_dim, device=torch.device("cpu"))
        self.register_buffer("pe", pe)  # Shape: (max_seq_len * 2, embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.start_embed = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.start_embed, mean=0.0, std=1.0 / math.sqrt(embed_dim))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

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

    def _interleave_start_embeddings(self, token_embeddings):
        B, T, C = token_embeddings.shape
        start = self.start_embed.view(1, 1, C).expand(B, T, C)
        stacked = torch.stack((token_embeddings, start), dim=2)
        return stacked.view(B, T * 2, C)

    def forward(self, idx, return_all_logits=None):
        if return_all_logits is None:
            return_all_logits = self.default_return_all_logits

        B, T = idx.shape
        token_embeddings = self.token_emb(idx)
        x = self._interleave_start_embeddings(token_embeddings)

        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds positional embedding limit {self.pe.size(0)}. "
                "Increase max_sequence_length in the config."
            )
        pos_embeddings = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pos_embeddings

        fixed_token_inputs = x[:, ::2, :].clone()

        attn_mask = self._build_attention_mask(T, device=idx.device)

        all_logits = [] if return_all_logits else None
        last_logits = None
        for _ in range(self.n_recursions):
            x = x.clone()
            x[:, ::2, :] = fixed_token_inputs
            for block in self.layers:
                x = block(x, attention_mask=attn_mask)

            normalized = self.ln_f(x)
            logits = self.head(normalized)
            if return_all_logits:
                all_logits.append(logits)
            last_logits = logits
            x = normalized

        if return_all_logits:
            return all_logits
        if last_logits is None:
            raise RuntimeError("No logits produced during recursion. Check n_recursions > 0.")
        return last_logits
