import torch
import torch.nn as nn
import chess
import torch.nn.functional as F
from dataclasses import dataclass

from .mod_transformer import Transformer
from .config import ModelConfig
from .model_base import BaseChet


class Chet_A(BaseChet):
    """
    Complete transformer model for chess move prediction.

    Args:
        embed_dim (int): Embedding dimension
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        dropout (float): Dropout probability (default: 0.1)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        self.config = config
        self.transformer = Transformer(config)
        self.move_predictor = MovePredictor(config.embed_dim)

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        """
        Predicts move probabilities for each piece on the board.

        Args:
            board_tokens (torch.Tensor): Tokenized board of shape [batch_size, 65]
        """

        batch_size = board_tokens.size(0)

        x = self.transformer(board_tokens)

        cls_embedding = x[:, 65, :]  # [batch_size, embed_dim]
        piece_embeddings = x[:, 0:64, :]  # [batch_size, 64, embed_dim]

        # p(from_square = i), p(to_square = j | from_square = i)
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
        temperature: float = 1.0
    ) -> list[tuple[chess.Move, float]]:
        with torch.no_grad():
            move_logits = self(board_tokens)  # [1, 4096]
            move_probs = F.softmax(move_logits / temperature, dim=-1)[0]  # [4096]

        moves_with_probs: list[tuple[chess.Move, float]] = []

        for move in board.legal_moves:
            # always promote to queen
            if move.promotion:
                move.promotion = chess.QUEEN

            from_square = move.from_square
            to_square = move.to_square
            idx = 64 * from_square + to_square
            prob = move_probs[idx].item()
            moves_with_probs.append((move, prob))

        moves_with_probs.sort(key=lambda x: x[1], reverse=True)
        return moves_with_probs[:n]


class MovePredictor(nn.Module):
    """
    Feed-forward network that predicts `P(from_square = i, to_square = j) = P(from_square = i) * P(to_square = j | from_square = i)`
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim

        # predicts P(from_square = i)
        self.from_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # predicts P(to_square = j | from_square = i)
        self.to_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
        )

    def forward(
        self, x: torch.Tensor, cls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts move scores for pieces.

        Args:
            x (torch.Tensor): Piece embeddings of shape [batch_size, 64, embed_dim]
            cls (torch.Tensor): Class token of shape [batch_size, embed_dim]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - From square scores of shape [batch_size, 64]
                - To square scores of shape [batch_size, 64, 64]
        """
        # Expand cls token to match x's sequence length
        batch_size = x.size(0)
        expanded_cls = cls.unsqueeze(1).expand(-1, 64, -1)

        # Concatenate expanded cls token with x
        x = torch.cat([x, expanded_cls], dim=-1)

        # Get from and to scores
        from_scores = self.from_head(x).squeeze(-1)  # [batch_size, 64]
        to_scores = self.to_head(x)  # [batch_size, 64, 64]

        return from_scores, to_scores


class Chet_A34(Chet_A):
    """
    Chet 34M model
    """

    def __init__(self) -> None:
        super().__init__(
            ModelConfig(embed_dim=480, n_heads=12, n_layers=12, dropout=0.1)
        )
