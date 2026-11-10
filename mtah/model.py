"""Set encoder per frame -> bidirectional temporal model -> H/A/D logits per frame.

Frame encoder: the ball token and up to P player tokens go through a small transformer (no
positional encoding: the players are a set). The ball token's output is the frame embedding,
concatenated with a masked mean of the player outputs.

Temporal model: dilated residual 1-D convs (local timing: touches, passes) followed by a
bidirectional GRU (context: a possession averages ~14 s). The whole window is seen at once, which
is fine because the task is offline.
"""

import torch
from torch import nn

from .features import F_BALL, F_PLAYER


def mlp(i, o, h=None):
    h = h or o
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class FrameEncoder(nn.Module):
    def __init__(self, d=128, heads=4, layers=2):
        super().__init__()
        self.player = mlp(F_PLAYER, d)
        self.ball = mlp(F_BALL, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.attn = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(2 * d, d)

    def forward(self, players, pmask, ball):
        B, L, P, _ = players.shape
        tok = torch.cat([self.ball(ball).unsqueeze(2), self.player(players)], dim=2)
        tok = tok.reshape(B * L, P + 1, -1)
        valid = torch.cat([torch.ones(B, L, 1, dtype=torch.bool, device=pmask.device), pmask], 2)
        valid = valid.reshape(B * L, P + 1)
        h = self.attn(tok, src_key_padding_mask=~valid)
        w = valid[:, 1:].unsqueeze(-1).to(h.dtype)
        pooled = (h[:, 1:] * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.out(torch.cat([h[:, 0], pooled], dim=-1)).reshape(B, L, -1)


class TemporalBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                       # (B, L, d)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + nn.functional.gelu(self.norm(y))


class PossessionNet(nn.Module):
    def __init__(self, d=128, heads=4, frame_layers=2, dilations=(1, 2, 4, 8, 16), gru_layers=2):
        super().__init__()
        self.config = dict(d=d, heads=heads, frame_layers=frame_layers,
                           dilations=list(dilations), gru_layers=gru_layers)
        self.frame = FrameEncoder(d, heads, frame_layers)
        self.tcn = nn.Sequential(*[TemporalBlock(d, k) for k in dilations])
        self.gru = nn.GRU(d, d, gru_layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 3))

    def forward(self, players, pmask, ball):
        x = self.frame(players, pmask, ball)
        x = self.tcn(x)
        x, _ = self.gru(x)
        return self.head(x)                      # (B, L, 3) logits for H, A, D
