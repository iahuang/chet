"""
Chet v3b — Architecture improvements over v3 at matched parameter count.

Changes from v3:
  - SwiGLU FFN replaces GELU FFN.  Hidden dim adjusted to 8/3 * embed_dim
    (rounded to nearest 64) so per-layer parameter count is unchanged.
  - RMSNorm replaces LayerNorm everywhere (pre-norm positions + final norm).

All other components (embedder, transformer body, cross-attention move head)
are identical to v3.
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
# Embedders (unchanged from v3)
# ---------------------------------------------------------------------------


class PieceEmbedder(nn.Module):
    """Embeds chess pieces into a continuous vector space."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).float()

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
    """Combines piece and positional embeddings for the 66-token board input."""

    def __init__(self, *, embed_dim: int) -> None:
        super().__init__()
        VOCAB_SIZE = 16
        self.piece_embedder = PieceEmbedder(VOCAB_SIZE, embed_dim)
        self.pos_embedder = PositionalEmbedding(embed_dim)

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = board_tokens.size(0)
        x = self.piece_embedder(board_tokens)
        pos_emb = self.pos_embedder(batch_size)
        x[:, :64, :] = x[:, :64, :] + pos_emb
        return x


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


class SwiGLU_FFN(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    Uses three projections with hidden_dim = round_up(8/3 * embed_dim, 64)
    so that total parameter count matches the standard 4x GELU FFN:
        Standard:  2 * embed_dim * (4 * embed_dim) = 8 * embed_dim^2
        SwiGLU:    3 * embed_dim * hidden_dim       ≈ 8 * embed_dim^2
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        hidden = int(embed_dim * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64  # round to 64 for hardware efficiency

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
        # Pre-norm attention
        x_attn = self.norm1(x)
        attd = self.attn(x_attn, x_attn, x_attn, need_weights=False)[0]
        x = x + attd

        # Pre-norm SwiGLU FFN
        x_ffn = self.norm2(x)
        ffn_out = self.ffn(x_ffn)
        x = x + ffn_out

        return x


# ---------------------------------------------------------------------------
# Move prediction head (unchanged from v3)
# ---------------------------------------------------------------------------


class MovePredictor(nn.Module):
    """Factored move prediction: from-head MLP + multi-head bilinear to-head."""

    def __init__(self, embed_dim: int, n_move_heads: int = 8) -> None:
        super().__init__()
        assert embed_dim % n_move_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_move_heads ({n_move_heads})"
        )
        self.n_heads = n_move_heads
        self.head_dim = embed_dim // n_move_heads
        self.scale = self.head_dim**-0.5

        # From-square head (per-square MLP with CLS context)
        self.from_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        # To-square head (multi-head bilinear Q·K scoring)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self, x: torch.Tensor, cls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # From-square scores
        expanded_cls = cls.unsqueeze(1).expand(-1, seq_len, -1)
        from_input = torch.cat([x, expanded_cls], dim=-1)  # [batch, 64, 2·D]
        from_scores = self.from_head(from_input).squeeze(-1)  # [batch, 64]

        # To-square scores via multi-head bilinear attention
        Q = self.query_proj(x)  # [batch, 64, D]
        K = self.key_proj(x)    # [batch, 64, D]

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        to_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        to_scores = to_scores.sum(dim=1)  # [batch, 64, 64]

        return from_scores, to_scores


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Chet(nn.Module):
    """Complete transformer model for chess move prediction (v3b).

    Architecture identical to v3 except:
      - SwiGLU FFN (matched param count)
      - RMSNorm everywhere
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

        # 1) Embed the board
        x = self.board_embedder(board_tokens)  # [batch, 66, embed_dim]

        # 2) Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # 3) Final normalization
        x = self.norm(x)

        # 4) Separate CLS token and 64 squares
        cls_embedding = x[:, 65, :]       # [batch, embed_dim]
        piece_embeddings = x[:, 0:64, :]  # [batch, 64, embed_dim]

        # 5) Predict moves
        from_logits, to_logits = self.move_predictor(piece_embeddings, cls_embedding)
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
            move_logits = self(board_tokens)  # [1, 4096]
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
