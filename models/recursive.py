from models.base import *

class RecursiveTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len, n_recursions=10):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.n_recursions = n_recursions

        pe = get_sinusoid_encoding_table(max_seq_len, embed_dim, device=torch.device("cpu"))
        self.register_buffer('pe', pe)  # Shape: (max_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        self.start_embed = nn.Parameter(torch.zeros(embed_dim)) # TODO: change to xavier init or something

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        token_embeddings = self.token_emb(idx)  # (B, T, embed_dim)

        start_embeds = torch.concat([self.start_embed for _ in range(B)]).unsqueeze(1)
        token_embeddings = torch.concat((token_embeddings, start_embeds), dim=1)

        torch.

        # repeat somehow token_embed,start_embed,token_embed,etc
        # then cross for loss e.g. each start_embed should turn into the token_embed
        # 2 options for the "junk" that gets produced from token_embed positions
        # 1) do loss as normal, promoting the model to become "identity" for "correct" answers
        # 2) dont use it

        # Use positional embeddings
        pos_embeddings = self.pe[:T, :].unsqueeze(0)  # (1, T, embed_dim)
        x = token_embeddings + pos_embeddings

        all_logits = []

        for n in range(self.n_recursions):
            for block in self.layers:
                x = block(x)

            x = self.ln_f(x)
            logits = self.head(x)  # (B, T, vocab_size)
            all_logits.append(logits)

        return all_logits