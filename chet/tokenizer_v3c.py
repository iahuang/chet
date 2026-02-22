"""
Tokenizer for Chet v3c — replaces the CLS token from v3b with a
repetition-count token so the model can reason about threefold-repetition
draws.  Sequence length stays at 66, matching v3b.

Token vocabulary (18 tokens):
    0      empty square
    1–6    white P, N, B, R, Q, K
    7–12   black p, n, b, r, q, k
    13     white to move
    14     black to move
    15     position seen 1× (no repetition)
    16     position seen 2× (one repetition — threefold is one move away)
    17     position seen 3+× (threefold repetition claimable)

Layout (66 tokens):
    [sq0 .. sq63]  [turn: 13|14]  [rep_count: 15|16|17]
"""

import chess
import torch


VOCAB_SIZE = 18
SEQ_LEN = 66

TOKEN_TURN_WHITE = 13
TOKEN_TURN_BLACK = 14

TOKEN_REP_1 = 15
TOKEN_REP_2 = 16
TOKEN_REP_3_PLUS = 17


def tokenize_board(board: chess.Board, repetition_count: int = 1) -> torch.Tensor:
    """Tokenize a board position into 66 tokens (v3c format).

    Args:
        board: The current board state.
        repetition_count: How many times this position has been seen in the
            game so far (1 = first time, 2 = one repetition, 3+ = draw
            claimable).  Callers replaying from a move list should track this
            themselves or use ``board.is_repetition()``.
    """
    tokens = torch.zeros(SEQ_LEN, dtype=torch.int)

    for i in range(64):
        tokens[i] = _piece_token(board.piece_at(i))

    tokens[64] = TOKEN_TURN_WHITE if board.turn == chess.WHITE else TOKEN_TURN_BLACK
    tokens[65] = _repetition_token(repetition_count)

    return tokens


def _repetition_token(count: int) -> int:
    if count <= 1:
        return TOKEN_REP_1
    if count == 2:
        return TOKEN_REP_2
    return TOKEN_REP_3_PLUS


_PIECE_TOKENS = {
    (chess.WHITE, chess.PAWN): 1,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4,
    (chess.WHITE, chess.QUEEN): 5,
    (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7,
    (chess.BLACK, chess.KNIGHT): 8,
    (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10,
    (chess.BLACK, chess.QUEEN): 11,
    (chess.BLACK, chess.KING): 12,
}


def _piece_token(piece: chess.Piece | None) -> int:
    if piece is None:
        return 0
    return _PIECE_TOKENS[(piece.color, piece.piece_type)]
