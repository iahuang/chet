from .model import Chet
from typing import Callable
import chess
import torch
import torch.nn.functional as F
import random


class TreeSearchStat:
    _sum_bf: int
    _sum_bf_reduction: float
    _n_branching_nodes: int
    n_internal_nodes: int
    n_terminal_nodes: int

    def __init__(self) -> None:
        self._sum_bf = 0
        self._sum_bf_reduction = 0
        self._n_branching_nodes = 0
        self.n_internal_nodes = 0
        self.n_terminal_nodes = 0

    def add_branching_node(self, branching_factor: int, bf_reduction: float) -> None:
        self._sum_bf += branching_factor
        self._sum_bf_reduction += bf_reduction
        self._n_branching_nodes += 1

    def add_terminal_node(self) -> None:
        self.n_terminal_nodes += 1

    def add_internal_node(self) -> None:
        self.n_internal_nodes += 1

    @property
    def avg_bf(self) -> float:
        return self._sum_bf / self._n_branching_nodes

    @property
    def avg_bf_reduction(self) -> float:
        return self._sum_bf_reduction / self._n_branching_nodes

    def __repr__(self) -> str:
        return f"TreeSearchStat(avg_bf={self.avg_bf:.2f}, avg_bf_reduction={self.avg_bf_reduction:.2f}, n_internal_nodes={self.n_internal_nodes}, n_terminal_nodes={self.n_terminal_nodes})"

    def __str__(self) -> str:
        return self.__repr__()


def _value_of_piece(piece: chess.Piece) -> int:
    if piece.piece_type == chess.PAWN:
        return 1
    elif piece.piece_type == chess.KNIGHT:
        return 3
    elif piece.piece_type == chess.BISHOP:
        return 3
    elif piece.piece_type == chess.ROOK:
        return 5
    elif piece.piece_type == chess.QUEEN:
        return 9

    return 0


def _simple_board_eval(board: chess.Board, side: chess.Color) -> float:
    """
    Evaluate the board using amodified version of Shannon's formula.

    f(board) = 9(Q - Q') + 5(R - R') + 3(B - B' + N - N') + 1(P - P')

    where Q, R, B, N, P are the number of queens, rooks, bishops, knights, and pawns respectively for
    `side`, and Q', R', B', N', P' are the number of queens, rooks, bishops, knights, and pawns respectively for
    the other side.
    """

    outcome = board.outcome()

    if outcome is not None:
        if outcome.winner == side:
            return 999
        elif outcome.winner is None:
            return 0.0
        else:
            return -999

    material = 0
    for _, piece in board.piece_map().items():
        if piece.color == side:
            material += _value_of_piece(piece)
        else:
            material -= _value_of_piece(piece)

    return material


def _get_next_moves(
    *,
    board: chess.Board,
    model: Chet,
    tokenizer: Callable[[chess.Board], torch.Tensor],
    gamma: float,
) -> list[tuple[chess.Move, float]]:
    """
    Return the subset of all legal moves that have a predicted probability / max(probabilities) > gamma.

    This is a simple heuristic to reduce the branching factor of the tree search.
    """

    assert 0 <= gamma <= 1

    with torch.no_grad():
        board_tokens = tokenizer(board).unsqueeze(0)
        move_logits = model(board_tokens)
        move_probs = F.softmax(move_logits, dim=-1)[0]

    moves_with_probs: list[tuple[chess.Move, float]] = []

    for move in board.legal_moves:
        if move.promotion:
            move.promotion = chess.QUEEN

        from_square = move.from_square
        to_square = move.to_square
        idx = 64 * from_square + to_square
        prob = move_probs[idx].item() / move_probs.max().item()

        if prob > gamma:
            moves_with_probs.append((move, prob))

    return moves_with_probs


def _minimax(
    *,
    board: chess.Board,
    side: chess.Color,
    depth: int,
    model: Chet,
    tokenizer: Callable[[chess.Board], torch.Tensor],
    gamma: float,
    alpha: float,
    beta: float,
    stat: TreeSearchStat,
) -> tuple[float, chess.Board]:
    """
    Return the minimax value of the board from the perspective of `side`, i.e. maximize the value
    of the child nodes when `board.turn == side` and minimize the value of the child nodes when
    `board.turn != side`.
    """

    # Base case: if we've reached the maximum depth or the game is over,
    # evaluate the position from the perspective of `side`
    if depth == 0 or board.is_game_over():
        stat.add_terminal_node()
        return _simple_board_eval(board, side), board.copy()

    # Get legal moves filtered by gamma threshold
    next_moves = _get_next_moves(
        board=board, model=model, tokenizer=tokenizer, gamma=gamma
    )
    stat.add_branching_node(
        len(next_moves), 1 - (len(next_moves) / len(list(board.legal_moves)))
    )
    stat.add_internal_node()
    # If no legal moves pass the gamma threshold, consider all legal moves
    if not next_moves:
        next_moves = [(move, 1.0) for move in board.legal_moves]

    if board.turn == side:
        # Maximizing player
        value = float("-inf")
        best_board = None
        for move, _ in next_moves:
            board.push(move)
            next_value, next_board = _minimax(
                board=board,
                side=side,
                depth=depth - 1,
                model=model,
                tokenizer=tokenizer,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
                stat=stat,
            )
            if next_value > value:
                value = next_value
                best_board = next_board
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        assert best_board is not None
        return value, best_board
    else:
        # Minimizing player
        value = float("inf")
        best_board = None
        for move, _ in next_moves:
            board.push(move)
            next_value, next_board = _minimax(
                board=board,
                side=side,
                depth=depth - 1,
                model=model,
                tokenizer=tokenizer,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
                stat=stat,
            )
            if next_value < value:
                value = next_value
                best_board = next_board
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cutoff
        assert best_board is not None
        return value, best_board


def chet_tree_search(
    *,
    model: Chet,
    tokenizer: Callable[[chess.Board], torch.Tensor],
    board: chess.Board,
    gamma: float,
    threshold: float,
    depth: int,
    temperature: float,
) -> tuple[chess.Move, float, TreeSearchStat, chess.Board]:
    """
    Perform a tree search using the given model to select the best move.

    - `gamma`: Child nodes are only considered if the move that makes them has `probability / max(probabilities) > gamma`.
    - `threshold`: If the best move by terminal evaluation has an absolute difference from the current evaluation greater than `threshold`, play that move.
                   Otherwise, if there are moves that worsen the evaluation by more than `threshold`, make sure they are not played.
    - `depth`: The maximum depth of the tree search.
    - `temperature`: The temperature to use for the move selection.
    """

    alpha = float("-inf")
    beta = float("inf")

    moves_with_terminal_values: list[tuple[chess.Move, float]] = []
    stat = TreeSearchStat()
    turn = board.turn

    for move in board.legal_moves:
        board.push(move)
        value, _ = _minimax(
            board=board,
            side=turn,
            depth=depth,
            model=model,
            tokenizer=tokenizer,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            stat=stat,
        )
        board.pop()
        moves_with_terminal_values.append((move, value))

    best_move_by_terminal_evals = max(moves_with_terminal_values, key=lambda x: x[1])
    current_eval = _simple_board_eval(board, turn)

    # if there is a move that improves the evaluation by more than threshold, play that move
    if best_move_by_terminal_evals[1] - current_eval > threshold:
        print(
            f"Playing move {best_move_by_terminal_evals[0]} (improves eval by {best_move_by_terminal_evals[1] - current_eval:.2f})"
        )
        return (
            best_move_by_terminal_evals[0],
            best_move_by_terminal_evals[1],
            stat,
            board,
        )

    # otherwise, if there are moves that worsen the evaluation by more than threshold, clamp their probabilities to 0
    logits = model(tokenizer(board).unsqueeze(0))

    for move, eval in moves_with_terminal_values:
        if current_eval - eval > threshold:
            idx = 64 * move.from_square + move.to_square
            logits[0, idx] = float("-inf")

    probs = F.softmax(logits / temperature, dim=-1)[0]

    # sample a move from the model's predicted probabilities
    moves_with_probs = []
    for move in board.legal_moves:
        if move.promotion:
            move.promotion = chess.QUEEN

        idx = 64 * move.from_square + move.to_square
        prob = probs[idx].item()
        moves_with_probs.append((move, prob))

    if sum(prob for _, prob in moves_with_probs) == 0:
        return random.choice(list(board.legal_moves)), current_eval, stat, board

    move = random.choices(
        moves_with_probs, weights=[prob for _, prob in moves_with_probs], k=1
    )[0][0]

    return move, current_eval, stat, board
