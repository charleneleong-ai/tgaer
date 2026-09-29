"""A small conv net over (board, action) predicting distance-to-win, fit per fold.

The value model's gradient-boosted baseline reads 21 global scalars — a colour
histogram, a centroid and a spread — so it can learn which action id is usually
good but never which is good *here*. This sees the board: 16 one-hot colour planes
plus a plane marking the clicked cell, so a click's context is spatial. Simple
actions carry their id through an embedding joined after pooling.

Global pooling makes the head independent of board size, and training data is a
few hundred rows per fold, so the net is deliberately small (~26k parameters).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

COLOURS = 16
ACTION_IDS = 8  # 0 is a click, whose cell rides in its own plane


def encode(grids: np.ndarray, prims: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(planes, action ids)`` for a stack of same-shaped boards."""
    boards = torch.as_tensor(np.asarray(grids), dtype=torch.long).clamp(0, COLOURS - 1)
    onehot = nn.functional.one_hot(boards, COLOURS).permute(0, 3, 1, 2).float()
    click = torch.zeros(len(prims), 1, *boards.shape[1:])
    ids = torch.zeros(len(prims), dtype=torch.long)
    for i, prim in enumerate(prims):
        if prim[0] == "click":
            click[i, 0, int(prim[1]), int(prim[2])] = 1.0
        else:
            ids[i] = int(prim[1])
    return torch.cat([onehot, click], 1), ids


class GridValue(nn.Module):
    """Three convs, global max pool, action embedding, scalar distance."""

    def __init__(self) -> None:
        super().__init__()
        width = 32
        self.conv = nn.Sequential(
            # Full resolution first: striding here loses single-cell positions.
            nn.Conv2d(COLOURS + 1, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.action = nn.Embedding(ACTION_IDS, width)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1)
        )

    def forward(self, planes: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        pooled = self.conv(planes).flatten(1)
        return self.head(torch.cat([pooled, self.action(ids)], 1)).squeeze(1)


def fit_predict(
    train_grids: np.ndarray,
    train_prims: list[tuple],
    train_y: np.ndarray,
    test_grids: np.ndarray,
    test_prims: list[tuple],
    *,
    steps: int = 600,
) -> np.ndarray:
    """Train from scratch on one fold and predict the held-out level's distances.

    A fixed step budget rather than fixed epochs, so a large fold costs no more
    than a small one.
    """
    torch.manual_seed(0)
    planes, ids = encode(train_grids, train_prims)
    y = torch.as_tensor(train_y, dtype=torch.float32)
    mean, sd = y.mean(), y.std(correction=0).clamp_min(1.0)
    target = (y - mean) / sd
    model = GridValue()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        batch = torch.randint(len(y), (64,))
        loss = nn.functional.mse_loss(model(planes[batch], ids[batch]), target[batch])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(*encode(test_grids, test_prims)) * sd + mean
    return pred.numpy()
