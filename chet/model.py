import torch
import torch.nn as nn
import chess
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    embed_dim: int
    n_heads: int
    n_layers: int
    dropout: float

    def as_dict(self):
        return {
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


class PieceEmbedder(nn.Module):
    """
    Embeds chess pieces into a continuous vector space.

    Args:
        vocab_size (int): Size of the piece vocabulary (number of unique piece types)
        embedding_dim (int): Dimension of the embedding vectors

    Attributes:
        embedding (nn.Embedding): Embedding layer that maps piece IDs to vectors
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of piece IDs into vectors.

        Args:
            x (torch.Tensor): Tensor of piece IDs

        Returns:
            torch.Tensor: Tensor of piece embeddings
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """
    Generates learnable positional embeddings for chess board squares.

    Each square's embedding is the sum of its rank (row) and file (column) embeddings.
    The embeddings are learned parameters that encode spatial relationships between squares.

    Args:
        embed_dim (int): Dimension of the embedding vectors

    Attributes:
        rank_embed (nn.Embedding): Embedding layer for ranks (0-7)
        file_embed (nn.Embedding): Embedding layer for files (0-7)
        positions (torch.Tensor): Buffer storing indices 0-63 for all squares
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.rank_embed = nn.Embedding(8, embed_dim)
        self.file_embed = nn.Embedding(8, embed_dim)

        positions = torch.arange(64).unsqueeze(0)  # shape [1, 64]
        self.register_buffer("positions", positions)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate positional embeddings for a batch of boards.

        Args:
            batch_size (int): Number of boards in the batch

        Returns:
            torch.Tensor: Positional embeddings with shape [batch_size, 64, embed_dim]
        """
        positions = self.positions.expand(batch_size, -1)  # shape [batch, 64]
        ranks = positions // 8
        files = positions % 8

        rank_emb = self.rank_embed(ranks)  # [batch, 64, embed_dim]
        file_emb = self.file_embed(files)  # [batch, 64, embed_dim]
        return rank_emb + file_emb


class BoardEmbedder(nn.Module):
    """
    Embeds chess board tokens into a learned representation by combining piece and positional embeddings.

    Args:
        embed_dim (int): Dimension of the embedding vectors
        n_heads (int): Number of attention heads (unused)
        n_layers (int): Number of transformer layers (unused)

    Attributes:
        piece_embedder (PieceEmbedder): Embedding layer for chess pieces and special tokens
        pos_embedder (PositionalEmbedding): Embedding layer for board positions
    """

    def __init__(self, *, embed_dim: int) -> None:
        super().__init__()

        VOCAB_SIZE = 16

        self.piece_embedder = PieceEmbedder(VOCAB_SIZE, embed_dim)
        self.pos_embedder = PositionalEmbedding(embed_dim)

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        """
        Generate embeddings for a batch of chess board tokens.

        Args:
            board_tokens (torch.Tensor): Tensor of shape [batch_size, 65] containing tokenized chess boards

        Returns:
            torch.Tensor: Combined piece and positional embeddings with shape [batch_size, 65, embed_dim]
        """

        batch_size = board_tokens.size(0)
        x = self.piece_embedder(board_tokens)

        # Only add positional embeddings to the board squares (first 64 tokens)
        pos_emb = self.pos_embedder(batch_size)
        x[:, :64, :] = x[:, :64, :] + pos_emb

        return x


class AttnFFN(nn.Module):
    """
    Feed-forward network used in transformer layers.

    Args:
        embed_dim (int): Input and output dimension

    Attributes:
        fc1 (nn.Linear): First linear layer
        fc2 (nn.Linear): Second linear layer
        activation (nn.LeakyReLU): Activation function
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        hidden_size = int(embed_dim * 4)

        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies feed-forward transformation to input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor
        """
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))

        return x


class TransformerLayer(nn.Module):
    """
    Single transformer layer with multi-head attention and feed-forward network.

    Args:
        embed_dim (int): Dimension of input/output embeddings
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability

    Attributes:
        attn (nn.MultiheadAttention): Multi-head attention layer
        ffn (AttnFFN): Feed-forward network
        norm1 (nn.LayerNorm): Layer normalization before attention
        norm2 (nn.LayerNorm): Layer normalization before feed-forward
        dropout (nn.Dropout): Dropout layer
        embed_dim (int): Embedding dimension
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True, dropout=dropout
        )
        self.ffn = AttnFFN(embed_dim)

        # Separate LayerNorms for attention and FFN (pre-norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Add dropout layers
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies transformer layer to input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor after attention and feed-forward
        """
        # Pre-norm before attention
        x_attn = self.norm1(x)
        attd = self.attn(x_attn, x_attn, x_attn, need_weights=False)[0]
        x = x + self.dropout(attd)  # Residual connection after attention with dropout

        # Pre-norm before FFN
        x_ffn = self.norm2(x)
        ffn_out = self.ffn(x_ffn)
        x = x + self.dropout(ffn_out)  # Residual connection after FFN with dropout

        return x


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


class Chet(nn.Module):
    """
    Complete transformer model for chess move prediction.

    Args:
        embed_dim (int): Embedding dimension
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        dropout (float): Dropout probability (default: 0.1)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        embed_dim = config.embed_dim
        n_heads = config.n_heads
        n_layers = config.n_layers
        dropout = config.dropout

        self.board_embedder = BoardEmbedder(embed_dim=embed_dim)

        self.embed_dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )

        self.move_predictor = MovePredictor(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        """
        Predicts move probabilities for each piece on the board.

        Args:
            board_tokens (torch.Tensor): Tokenized board of shape [batch_size, 65]
            legal_moves_mask (torch.Tensor): Mask of shape [batch_size, 4096] (optional)
            where the move `i = 64 * from_square + to_square` is legal if
            `legal_moves_mask[b, i] = 1`

        Returns:
            torch.Tensor: Move logits of shape [batch_size, 4096] where the `64i + j`th index
            along the last dimension is the logit of the move from `i` to `j`
        """
        batch_size = board_tokens.size(0)

        # 1) Embed the board and apply dropout
        x = self.board_embedder(board_tokens)  # [batch_size, 66, embed_dim]
        x = self.embed_dropout(x)

        # 2) Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # 3) Final normalization
        x = self.norm(x)

        # 4) Separate the CLS token and the 64 squares
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
        """
        Get the top n legal moves predicted by the model for a given board position.

        Args:
            board_tokens (torch.Tensor): Tokenized board of shape [1, 65]
            board (chess.Board): The chess board state to get moves for
            n (int): Number of top moves to return. Defaults to 5.

        Returns:
            list[tuple[chess.Move, float]]: List of (move, probability) tuples for the top n legal moves
        """
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

    def get_n_params(self):
        """
        Get the number of trainable parameters in the model.
        """

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, path: str, config: ModelConfig, *, device: str = "cpu"):
        """
        Load a pretrained model from a file.

        Args:
            path (str): The path to the model file
            config (ModelConfig): The configuration used to train the model
            device (str): The device to load the model onto
        """

        model = cls(config)
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device)
