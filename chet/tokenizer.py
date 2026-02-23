import chess
import torch


def tokenize_board(board: chess.Board) -> torch.Tensor:
    """Tokenize a chess board into a 65-element tensor.

    Tokens:
        0:     empty square
        1-6:   white P, N, B, R, Q, K  (matches chess.PAWN..chess.KING)
        7-12:  black P, N, B, R, Q, K
        13:    white to move
        14:    black to move

    Layout: [sq0 .. sq63] [turn token]
    Square indices follow python-chess convention: a1=0, b1=1, ..., h8=63.
    """
    tokens = torch.zeros(65, dtype=torch.int)

    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            tokens[i] = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6

    tokens[64] = 13 if board.turn == chess.WHITE else 14

    return tokens
