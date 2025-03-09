"""
Script for aggregating training data from PGN files acquired from https://database.nikonoel.fr/
Data files themselves are not included in the repository due to size.
"""

import chess
from chess import pgn
from tqdm import tqdm
from typing import TextIO
import time

MOVE_INDEX_CHECK = 20
MAX_FREQ_OF_POSITION = 1 / 1000
PRUNE_MIN_FREQ = 1 / 10000
PRINT_EVERY_N_GAMES = 100
PRUNE_EVERY_N_GAMES = 10000


class PositionCounter:
    """
    Counts the number of times a position has occurred in a given dataset.
    """

    _ocurrences: dict[str, int]

    def __init__(self) -> None:
        self._ocurrences = {}

    def increment(self, fen: str) -> None:
        if fen not in self._ocurrences:
            self._ocurrences[fen] = 0

        self._ocurrences[fen] += 1

    def get_count(self, fen: str) -> int:
        return self._ocurrences.get(fen, 0)

    def get_total_count(self) -> int:
        return sum(self._ocurrences.values())

    def length(self) -> int:
        return len(self._ocurrences)

    def prune(self, n_written: int) -> None:
        n_pruned = 0
        for fen, count in list(self._ocurrences.items()):
            if count / (n_written + 1) < PRUNE_MIN_FREQ:
                del self._ocurrences[fen]
                n_pruned += 1

        print(f"pruned {n_pruned} positions")

    def print_top_n_most_common(self, n: int, n_written: int) -> None:
        sorted_by_count = sorted(
            self._ocurrences.items(), key=lambda x: x[1], reverse=True
        )

        for fen, count in sorted_by_count[:n]:
            freq = count / (n_written + 1)
            print(f"{fen} count: {count} freq: {freq:.2%}")


def fmt_time(seconds: float) -> str:
    seconds = round(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def write_examples_from_pgn(
    pgn_file: str, output_file: TextIO, counter: PositionCounter
) -> None:
    """
    Write examples from a PGN file to an output file.
    """

    n_skipped = 0
    n_written = 0
    n_games = 0

    start_time = time.time()

    with open(pgn_file, "r") as f:
        total_games = f.read().count("\n1. ")

    with open(pgn_file, "r") as f:
        while True:
            game = pgn.read_game(f)

            if game is None:
                break

            board = game.board()

            move_index = 0
            for move in game.mainline_moves():
                fen = board.fen()

                skip = False

                # for positions which occur early in the game, we want to
                # ensure that they do not appear too frequently in the dataset.
                # for memory efficiency, we do not bother checking the frequency
                # of positions which occur later in the game, since they are almost
                # certainly not too common.
                if move_index < MOVE_INDEX_CHECK:
                    n_instances = counter.get_count(fen)
                    freq = n_instances / (n_written + 1)

                    if freq > MAX_FREQ_OF_POSITION:
                        skip = True

                if not skip:
                    counter.increment(fen)
                    output_file.write(f"{fen},{move.uci()}\n")
                    n_written += 1
                else:
                    n_skipped += 1

                board.push(move)
                move_index += 1

            n_games += 1

            if n_games % PRINT_EVERY_N_GAMES == 0 and n_games > 0:
                remaining_games = total_games - n_games
                rate = n_games / (time.time() - start_time)
                remaining_time = remaining_games / rate
                print(
                    f"# games processed: {n_games:,}, # written: {n_written:,}, # skipped: {n_skipped:,}, % skipped: {n_skipped / (n_written + n_skipped):.2%}, # positions in counter: {counter.length():,}, progress: {n_games / total_games:.2%}, games/sec: {rate:.2f}, remaining: {fmt_time(remaining_time)}"
                )
                # counter.print_top_n_most_common(100, n_written)

            if n_games % PRUNE_EVERY_N_GAMES == 0 and n_games > 0:
                counter.prune(n_written)


def write_examples_from_puzzles(puzzle_file: str, output_file: TextIO) -> None:
    """
    Write examples from a puzzle file to an output file.
    """

    with open(puzzle_file, "r") as f:
        data = f.readlines()

    total = len(data)
    n_written = 0
    n_processed = 0

    start_time = time.time()

    for d in data[1:]:
        row = d.split(",")
        fen = row[1]
        moves = row[2].split(" ")

        board = chess.Board(fen)

        j = 0
        for move in moves:
            # only write the puzzle moves, not the moves played in response
            if j % 2 == 1:
                output_file.write(f"{board.fen()},{move}\n")
                n_written += 1
            board.push_uci(move)
            j += 1

        n_processed += 1

        if n_processed % 1000 == 0:
            rate = n_processed / (time.time() - start_time) 
            remaining_time = (total - n_processed) / rate
            print(f"# puzzles processed: {n_processed:,}, # written: {n_written:,}, progress: {n_processed / total:.2%}, puzzles/sec: {rate:.2f}, remaining: {fmt_time(remaining_time)}")


# with open("training_data.csv", "w") as f:
#     counter = PositionCounter()
#     write_examples_from_pgn("lichess_elite_2022-01.pgn", f, counter)
#     write_examples_from_pgn("lichess_elite_2022-02.pgn", f, counter)
#     write_examples_from_pgn("lichess_elite_2022-03.pgn", f, counter)
#     write_examples_from_pgn("lichess_elite_2022-04.pgn", f, counter)

with open("training_data_2.csv", "w") as f:
    write_examples_from_puzzles("lichess_db_puzzle.csv", f)
