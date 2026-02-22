"""
Chet v3c — Extends v3b with threefold-repetition awareness.

Changes from v3b:
  - CLS token removed; replaced by a repetition-count token at the same
    position (index 65).  Sequence length stays at 66.
  - Vocab 16 → 18: old CLS (15) is repurposed as rep×1 (15), plus two new
    tokens rep×2 (16) and rep×3+ (17).
  - Move-prediction from-head no longer concatenates a global context vector;
    it operates on per-square embeddings only (the transformer already fuses
    global context via self-attention over the turn and repetition tokens).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from dataclasses import dataclass


@dataclass
class ModelConfig:
    embed_dim: int
    n_heads: int
    n_layers: int
    n_move_heads: int = 8

    def as_dict(self):
        return {
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "n_move_heads": self.n_move_heads,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class PieceEmbedder(nn.Module):
    """Embeds chess tokens (pieces + special tokens) into a continuous vector space."""

    VOCAB_SIZE = 18  # 0–12 pieces, 13–14 turn, 15–17 repetition

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(self.VOCAB_SIZE, embedding_dim).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings for 64 board squares."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, embed_dim)
        positions = torch.arange(64).unsqueeze(0)  # [1, 64]
        self.register_buffer("positions", positions)

    def forward(self, batch_size: int) -> torch.Tensor:
        positions = self.positions.expand(batch_size, -1)  # [batch, 64]
        return self.embed(positions)  # [batch, 64, embed_dim]


class BoardEmbedder(nn.Module):
    """Combines piece and positional embeddings for the 66-token board input.

    Token layout:
        [sq0..sq63]  [turn]  [rep_count]
         indices 0–63   64       65

    Only the 64 square tokens receive positional embeddings.
    The repetition-count token at index 65 serves as the global context
    vector (replacing the CLS token from v3b).
    """

    def __init__(self, *, embed_dim: int) -> None:
        super().__init__()
        self.piece_embedder = PieceEmbedder(embed_dim)
        self.pos_embedder = PositionalEmbedding(embed_dim)

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = board_tokens.size(0)
        x = self.piece_embedder(board_tokens)       # [batch, 66, embed_dim]
        pos_emb = self.pos_embedder(batch_size)      # [batch, 64, embed_dim]
        x[:, :64, :] = x[:, :64, :] + pos_emb
        return x


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


class SwiGLU_FFN(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    Uses three projections with hidden_dim = round_up(8/3 * embed_dim, 64)
    so that total parameter count matches the standard 4x GELU FFN.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        hidden = int(embed_dim * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64

        self.w1 = nn.Linear(embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer layer (SwiGLU + RMSNorm)
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with multi-head attention and SwiGLU FFN."""

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True
        )
        self.ffn = SwiGLU_FFN(embed_dim)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn = self.norm1(x)
        attd = self.attn(x_attn, x_attn, x_attn, need_weights=False)[0]
        x = x + attd

        x_ffn = self.norm2(x)
        ffn_out = self.ffn(x_ffn)
        x = x + ffn_out

        return x


# ---------------------------------------------------------------------------
# Move prediction head
# ---------------------------------------------------------------------------


class MovePredictor(nn.Module):
    """Factored move prediction: from-head MLP + multi-head bilinear to-head.

    Unlike v3b, global context (repetition token) is not concatenated into the
    from-head — the transformer already fuses that information into the
    per-square representations via self-attention.
    """

    def __init__(self, embed_dim: int, n_move_heads: int = 8) -> None:
        super().__init__()
        assert embed_dim % n_move_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_move_heads ({n_move_heads})"
        )
        self.n_heads = n_move_heads
        self.head_dim = embed_dim // n_move_heads
        self.scale = self.head_dim**-0.5

        self.from_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        from_scores = self.from_head(x).squeeze(-1)  # [batch, 64]

        Q = self.query_proj(x)
        K = self.key_proj(x)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        to_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        to_scores = to_scores.sum(dim=1)  # [batch, 64, 64]

        return from_scores, to_scores


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Chet(nn.Module):
    """Complete transformer model for chess move prediction (v3c).

    Architecture identical to v3b except:
      - CLS token replaced by repetition-count token at index 65 (same
        sequence length of 66)
      - Vocab size 18 (rep tokens at 15, 16, 17 replace old CLS at 15)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        embed_dim = config.embed_dim
        n_heads = config.n_heads
        n_layers = config.n_layers
        n_move_heads = config.n_move_heads

        self.board_embedder = BoardEmbedder(embed_dim=embed_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(embed_dim, n_heads) for _ in range(n_layers)]
        )
        self.move_predictor = MovePredictor(embed_dim, n_move_heads=n_move_heads)
        self.norm = RMSNorm(embed_dim)

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = board_tokens.size(0)

        x = self.board_embedder(board_tokens)  # [batch, 66, embed_dim]

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)

        piece_embeddings = x[:, 0:64, :]  # [batch, 64, embed_dim]

        from_logits, to_logits = self.move_predictor(piece_embeddings)
        full_logits = from_logits.unsqueeze(-1) + to_logits
        full_logits = full_logits.view(batch_size, 64 * 64)

        return full_logits

    def get_top_moves(
        self,
        board_tokens: torch.Tensor,
        board: chess.Board,
        n: int = 5,
        *,
        temperature: float = 1.0,
    ) -> list[tuple[chess.Move, float]]:
        with torch.no_grad():
            move_logits = self(board_tokens)
            move_probs = F.softmax(move_logits / temperature, dim=-1)[0]

        moves_with_probs: list[tuple[chess.Move, float]] = []
        for move in board.legal_moves:
            if move.promotion:
                move.promotion = chess.QUEEN
            from_square = move.from_square
            to_square = move.to_square
            idx = 64 * from_square + to_square
            prob = move_probs[idx].item()
            moves_with_probs.append((move, prob))

        moves_with_probs.sort(key=lambda x: x[1], reverse=True)
        return moves_with_probs[:n]

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, path: str, config: ModelConfig, *, device: str = "cpu"):
        model = cls(config)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model.to(device)
