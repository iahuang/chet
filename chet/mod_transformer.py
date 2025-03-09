import torch
import torch.nn as nn

from .config import ModelConfig
from .mod_embedder import BoardEmbedder


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


class Transformer(nn.Module):
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

        return x
