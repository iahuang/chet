from __future__ import annotations
import chess
import torch
import torch.nn.functional as F
from chet.model import Chet
from typing import Callable
import math
import random


class GameTreeNode:
    """
    A node in the game tree.
    """

    # The current board state
    board: chess.Board

    # The legal moves from the current board state
    legal_moves: list[chess.Move]

    # The log probability of the this path being played from the root node
    # (i.e. the sum of `path_log_prob` for all nodes on the path from the root to this node)
    path_log_prob: float

    # The probability of this node being played from the root node
    self_prob: float

    # The depth of this node in the tree
    depth: int

    # The minimax value of this node
    value: float | None

    # The children of this node
    children: dict[chess.Move, GameTreeNode]

    def __init__(
        self,
        *,
        board: chess.Board,
        legal_moves: list[chess.Move],
        path_log_prob: float,
        depth: int,
        prob: float,
    ) -> None:
        self.board = board
        self.legal_moves = legal_moves
        self.path_log_prob = path_log_prob
        self.depth = depth
        self.value = None
        self.children = {}
        self.self_prob = prob

    def get_value(self):
        assert self.value is not None
        return self.value

    def _get_leaves(self, leaves: list[GameTreeNode]) -> list[GameTreeNode]:
        if self.children:
            for child in self.children.values():
                child._get_leaves(leaves)
        else:
            leaves.append(self)
        return leaves

    def get_leaves(self) -> list[GameTreeNode]:
        leaves = []
        return self._get_leaves(leaves)

    def is_game_state_terminal(self) -> bool:
        return self.board.is_game_over()

    def create_child(self, move: chess.Move, prob: float) -> GameTreeNode:
        child = GameTreeNode(
            board=self.board.copy(),
            legal_moves=self.legal_moves,
            path_log_prob=self.path_log_prob + math.log(prob),
            depth=self.depth + 1,
            prob=prob,
        )

        child.board.push(move)
        child.legal_moves = list(child.board.legal_moves)
        self.children[move] = child
        return child

    def compute_minimax_values(self, side: chess.Color) -> None:
        if self.is_game_state_terminal() or not self.children:
            self.value = _simple_board_eval(self.board, side)
        else:
            for child in self.children.values():
                child.compute_minimax_values(side)

            if self.board.turn == side:
                self.value = max(child.get_value() for child in self.children.values())
            else:
                self.value = min(child.get_value() for child in self.children.values())

    @classmethod
    def create_root(
        cls,
        board: chess.Board,
    ) -> GameTreeNode:
        return cls(
            board=board,
            legal_moves=list(board.legal_moves),
            path_log_prob=0,
            depth=0,
            prob=1,
        )


def _tree_expand(
    *,
    root: GameTreeNode,
    model: Chet,
    tokenizer: Callable[[chess.Board], torch.Tensor],
    gamma: float,
    limit: int | None = None,
) -> list[GameTreeNode]:
    leaves = [leaf for leaf in root.get_leaves() if not leaf.is_game_state_terminal()]

    if len(leaves) == 0:
        return []

    if limit is not None:
        leaves.sort(key=lambda leaf: leaf.path_log_prob, reverse=True)
        leaves = leaves[:limit]

    batch = torch.stack([tokenizer(leaf.board) for leaf in leaves])

    with torch.no_grad():
        batch_logits = model(batch)
        batch_probs = F.softmax(batch_logits, dim=-1)

    i = 0
    for leaf in leaves:
        for move in leaf.legal_moves:
            if move.promotion:
                move.promotion = chess.QUEEN

            from_square = move.from_square
            to_square = move.to_square
            idx = 64 * from_square + to_square

            rel_prob = batch_probs[i, idx].item() / batch_probs[i].max().item()

            board = leaf.board.copy()
            board.push(move)
            
            if (
                rel_prob > gamma
                or board.is_check()
                or board.is_game_over()
                or board.is_stalemate()
            ):
                leaf.create_child(move, rel_prob)

        i += 1

    return leaves


def get_move(
    *,
    board: chess.Board,
    model: Chet,
    tokenizer: Callable[[chess.Board], torch.Tensor],
    gamma: float,
    depth: int,
    temperature: float,
    threshold: float = 1,
) -> chess.Move:
    root = GameTreeNode.create_root(board)

    for _ in range(depth):
        _tree_expand(
            root=root,
            model=model,
            tokenizer=tokenizer,
            gamma=gamma,
        )

    root.compute_minimax_values(board.turn)

    # if there is a move that improves or worsens the evaluation by more than threshold, play that move
    best_move_by_eval = max(root.children.items(), key=lambda x: x[1].get_value())
    current_eval = _simple_board_eval(board, board.turn)

    if abs(best_move_by_eval[1].get_value() - current_eval) > threshold:
        print(
            f"Playing move {best_move_by_eval[0]} (improves eval by {best_move_by_eval[1].get_value() - current_eval:.2f})"
        )
        return best_move_by_eval[0]

    # otherwise, remove all moves that worsen the evaluation by more than threshold
    logits = model(tokenizer(board).unsqueeze(0))

    for move, child in root.children.items():
        if abs(child.get_value() - current_eval) > threshold:
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

    move = random.choices(
        moves_with_probs, weights=[prob for _, prob in moves_with_probs], k=1
    )[0][0]
    return move


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
