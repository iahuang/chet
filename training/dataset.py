from __future__ import annotations
import csv
from typing import Callable
import chess
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random


def load_dataset(
    csv_file: str,
    tokenizer: Callable[[chess.Board], torch.Tensor],
    *,
    limit: int | None = None,
    skip_header: bool = True,
) -> ChessDataset:
    """
    Load a dataset from a CSV file.

    Expected format:
    ```
    board_fen,move_uci
    rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4
    ...
    ```
    """

    boards = []
    moves = []

    with open(csv_file, "r") as f:
        reader = csv.reader(f)

        if skip_header:
            next(reader)

        for row in tqdm(reader, desc="Loading dataset", total=limit):
            board_fen, move_uci = row
            board = board_fen
            boards.append(board)

            move = move_uci
            moves.append(move)

            if limit and len(moves) == limit:
                break

    return ChessDataset(boards, moves, tokenizer)


class ChessDataset(Dataset):
    boards: list[str]
    moves: list[str]
    tokenizer: Callable[[chess.Board], torch.Tensor]

    def __init__(
        self,
        boards: list[str],
        moves: list[str],
        tokenizer: Callable[[chess.Board], torch.Tensor],
    ) -> None:
        super().__init__()
        self.boards = boards
        self.moves = moves
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training example from the dataset.

        Args:
            idx (int): Index of the example to get

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Board tokens tensor of shape [65]
                - Target move probabilities tensor of shape [4096]
                - Legal move mask tensor of shape [4096]. 1.0 if the move is legal, 0.0 otherwise.
        """

        board = chess.Board(self.boards[idx])
        move = chess.Move.from_uci(self.moves[idx])

        # Convert board to tokens
        board_tokens = self.tokenizer(board)

        # Create target probability distribution
        target = torch.zeros(4096)

        target[move.from_square * 64 + move.to_square] = 1.0

        legal_move_mask = torch.zeros(4096)
        for move in board.legal_moves:
            legal_move_mask[move.from_square * 64 + move.to_square] = 1.0

        return board_tokens, target, legal_move_mask


def merge_datasets(datasets: list[ChessDataset]) -> ChessDataset:
    boards = []
    moves = []

    for dataset in datasets:
        boards.extend(dataset.boards)
        moves.extend(dataset.moves)

    return ChessDataset(boards, moves, datasets[0].tokenizer)


def shuffle_dataset(dataset: ChessDataset) -> ChessDataset:
    boards, moves = dataset.boards, dataset.moves
    indices = list(range(len(boards)))
    random.shuffle(indices)

    return ChessDataset(
        [boards[i] for i in indices],
        [moves[i] for i in indices],
        dataset.tokenizer,
    )


def split_dataset(
    dataset: ChessDataset, val_split: float
) -> tuple[ChessDataset, ChessDataset]:
    n_val = int(len(dataset) * val_split)
    train_boards = dataset.boards[:-n_val]
    val_boards = dataset.boards[-n_val:]

    train_moves = dataset.moves[:-n_val]
    val_moves = dataset.moves[-n_val:]

    return (
        ChessDataset(train_boards, train_moves, dataset.tokenizer),
        ChessDataset(val_boards, val_moves, dataset.tokenizer),
    )
