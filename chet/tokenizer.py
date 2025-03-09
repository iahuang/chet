import chess
import torch


def tokenize_board(board: chess.Board) -> torch.Tensor:
    """
    Tokenizes a chess board into a 66 element tensor.

    Tokens

    0: empty square
    1: white pawn
    2: white knight
    3: white bishop
    4: white rook
    5: white queen
    6: white king
    7: black pawn
    8: black knight
    9: black bishop
    10: black rook
    11: black queen
    12: black king
    13: [move token: white to move]
    14: [move token: black to move]

    Tokenization layout:
    ```
    0          1              63          64                    
    [square 0] [square 1] ... [square 63] [move token: 13 or 14]
    ```
    """

    tokenization = torch.zeros(65, dtype=torch.int)
    tokenization[64] = _move_tokenization(board.turn)

    for i in range(64):
        tokenization[i] = _piece_tokenization(board.piece_at(i))

    return tokenization


def _move_tokenization(turn: chess.Color) -> int:
    if turn == chess.WHITE:
        return 13
    else:
        return 14


def _piece_tokenization(piece: chess.Piece | None) -> int:
    if piece is None:
        return 0
    else:
        if piece.color == chess.WHITE:
            if piece.piece_type == chess.PAWN:
                return 1
            elif piece.piece_type == chess.KNIGHT:
                return 2
            elif piece.piece_type == chess.BISHOP:
                return 3
            elif piece.piece_type == chess.ROOK:
                return 4
            elif piece.piece_type == chess.QUEEN:
                return 5
            elif piece.piece_type == chess.KING:
                return 6
        else:
            if piece.piece_type == chess.PAWN:
                return 7
            elif piece.piece_type == chess.KNIGHT:
                return 8
            elif piece.piece_type == chess.BISHOP:
                return 9
            elif piece.piece_type == chess.ROOK:
                return 10
            elif piece.piece_type == chess.QUEEN:
                return 11
            elif piece.piece_type == chess.KING:
                return 12

    raise ValueError("Invalid piece")
