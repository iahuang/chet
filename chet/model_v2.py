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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


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
    """Combines piece and positional embeddings for the 65-token board input.

    Input layout: [sq0..sq63, turn_token].  Positional embeddings are added
    only to the 64 square tokens; the turn token carries no spatial position.
    """

    VOCAB_SIZE = 15

    def __init__(self, *, embed_dim: int) -> None:
        super().__init__()
        self.piece_embedder = PieceEmbedder(self.VOCAB_SIZE, embed_dim)
        self.pos_embedder = PositionalEmbedding(embed_dim)

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = board_tokens.size(0)
        x = self.piece_embedder(board_tokens)
        x[:, :64, :] = x[:, :64, :] + self.pos_embedder(batch_size)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with explicit Q/K/V/O projections.

    Uses F.scaled_dot_product_attention for automatic FlashAttention / memory-
    efficient kernel selection.
    """

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)


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


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with multi-head self-attention and SwiGLU FFN."""

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn = Attention(embed_dim, n_heads)
        self.ffn = SwiGLU_FFN(embed_dim)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Move prediction head
# ---------------------------------------------------------------------------


class MovePredictor(nn.Module):
    """Factored move prediction: from-head MLP + multi-head bilinear to-head.

    The from-head scores each square independently using its (already
    contextualized) embedding.  The to-head uses multi-head bilinear Q·K
    attention over square embeddings to score from→to pairs.
    """

    def __init__(self, embed_dim: int, n_move_heads: int = 8) -> None:
        super().__init__()
        assert (
            embed_dim % n_move_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by n_move_heads ({n_move_heads})"
        self.n_heads = n_move_heads
        self.head_dim = embed_dim // n_move_heads
        self.scale = self.head_dim**-0.5

        self.from_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, squares: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = squares.shape

        from_scores = self.from_head(squares).squeeze(-1)  # [batch, 64]

        Q = self.query_proj(squares).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(squares).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        to_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        to_scores = to_scores.sum(dim=1)  # [batch, 64, 64]

        return from_scores, to_scores


class Chet(nn.Module):
    """Transformer model for chess move prediction.

    Input: 65 tokens = 64 board squares + 1 turn indicator.
    Output: 4096 logits (64 from-squares x 64 to-squares).

    The turn token participates in self-attention so every square embedding is
    conditioned on whose move it is.  Only the 64 square embeddings are fed to
    the move prediction head.
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

        x = self.board_embedder(board_tokens)  # [batch, 65, embed_dim]

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)

        squares = x[:, :64, :]  # [batch, 64, embed_dim]

        from_logits, to_logits = self.move_predictor(squares)
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
