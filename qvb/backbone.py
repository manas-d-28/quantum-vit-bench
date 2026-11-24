"""The shared tiny ViT. Everything here is classical and identical for every model;
only the `attention` module passed in changes.

Block (standard pre-norm transformer):
    x = x + W_O · Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
Tokens -> linear embedding + learned positional embedding -> blocks ->
LayerNorm -> mean over tokens -> linear classifier.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, embed_dim, ff_dim, attention):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = attention                 # maps (B, T, E) -> (B, T, E)
        self.proj = nn.Linear(embed_dim, embed_dim)   # W_O, classical for every model
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(),
                                 nn.Linear(ff_dim, embed_dim))

    def forward(self, x):
        x = x + self.proj(self.attn(self.norm1(x)))
        return x + self.ffn(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, n_tokens, token_dim, make_attention, embed_dim=16, n_layers=2,
                 ff_dim=32, n_classes=10):
        super().__init__()
        self.embed = nn.Linear(token_dim, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([Block(embed_dim, ff_dim, make_attention())
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.embed(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))
