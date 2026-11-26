"""The shared tiny ViT. Everything here is classical and identical for every model;
only the `attention` module passed in changes. (Exception: Comajoan Cara's model also
makes the output projection W_O and a layer inside the FFN quantum, because that is
what their paper does; those are passed in via `make_proj` / `make_ffn`.)

Block (standard pre-norm transformer):
    x = x + W_O · Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
Tokens -> linear embedding + learned positional embedding -> blocks ->
LayerNorm -> mean over tokens -> linear classifier.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, embed_dim, ff_dim, attention, proj=None, ffn=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = attention                 # maps (B, T, E) -> (B, T, E)
        self.proj = proj if proj is not None else nn.Linear(embed_dim, embed_dim)   # W_O
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = ffn if ffn is not None else nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim))

    def forward(self, x):
        x = x + self.proj(self.attn(self.norm1(x)))
        return x + self.ffn(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, n_tokens, token_dim, make_attention, embed_dim=16, n_layers=2,
                 ff_dim=32, n_classes=10, make_proj=None, make_ffn=None):
        super().__init__()
        self.embed = nn.Linear(token_dim, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            Block(embed_dim, ff_dim, make_attention(),
                  make_proj() if make_proj else None, make_ffn() if make_ffn else None)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.embed(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))
