import torch.nn as nn, torch


class LogBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256,
                 n_heads=8, n_layers=8, max_len=40):
        super().__init__()
        #self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=MASK_IDX)
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim,
            batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlkp_head = nn.Linear(embed_dim, vocab_size)   # predict token id

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   mean=0, std=0.02)
        nn.init.normal_(self.mlkp_head.weight, mean=0, std=0.02)

    def forward(self, x):
        # x: [B, L]  integer ids
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.encoder(h)                     # [B, L, D]
        logits = self.mlkp_head(h)              # MLKP output
        h_dist = h[:, 0, :]                     # embedding at DIST pos
        return logits, h_dist                  # used for VHM
