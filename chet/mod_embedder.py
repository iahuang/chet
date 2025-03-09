import torch
import torch.nn as nn


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
