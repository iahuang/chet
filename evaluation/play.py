import random
import sys
import os

import chess
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chet.model import Chet, ModelConfig
from chet.tokenizer import tokenize_board


def play_game(
    white: Chet,
    black: Chet,
    device: str = "cpu",
    temperature: float = 1.0,
    max_moves: int = 300,
) -> str:
    """
    Play a single game between two models.

    Returns:
        "white", "black", or "draw"
    """
    board = chess.Board()

    for _ in range(max_moves):
        if board.is_game_over():
            break

        model = white if board.turn == chess.WHITE else black
        board_tensor = tokenize_board(board).unsqueeze(0).to(device)

        top_moves = model.get_top_moves(board_tensor, board, n=5, temperature=temperature)
        move = random.choices(top_moves, weights=[p for _, p in top_moves], k=1)[0][0]
        board.push(move)

    result = board.result()
    if result == "1-0":
        return "white"
    elif result == "0-1":
        return "black"
    return "draw"


def play_match(
    white: Chet,
    black: Chet,
    n_games: int,
    device: str = "cpu",
    temperature: float = 1.0,
    max_moves: int = 300,
) -> dict:
    """
    Play n_games between two models and return results.
    """
    results = {"white": 0, "black": 0, "draw": 0}

    for i in range(n_games):
        outcome = play_game(white, black, device, temperature, max_moves)
        results[outcome] += 1
        print_result_bar(results, n_games, i + 1)

    return results


def print_result_bar(results: dict, total: int, played: int):
    """
    Print a color-coded bar showing wins/draws/losses.
    """
    BAR_WIDTH = 40

    w = results["white"]
    b = results["black"]
    d = results["draw"]

    w_cells = round(w / played * BAR_WIDTH)
    b_cells = round(b / played * BAR_WIDTH)
    d_cells = BAR_WIDTH - w_cells - b_cells

    WHITE_BG = "\033[48;2;129;212;105m\033[30m"  # green bg
    BLACK_BG = "\033[48;2;235;87;87m\033[97m"    # red bg
    DRAW_BG = "\033[48;2;180;180;180m\033[30m"   # gray bg
    RESET = "\033[0m"

    bar = (
        WHITE_BG + (" " * w_cells) + RESET
        + DRAW_BG + (" " * d_cells) + RESET
        + BLACK_BG + (" " * b_cells) + RESET
    )

    summary = f"W {w}  D {d}  L {b}  ({played}/{total})"
    print(f"\r{bar}  {summary}", end="", flush=True)

    if played == total:
        print()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Define models here ---
    from chet.pretrained import chet_1_base
    from chet.model_v2 import Chet, ModelConfig

    white_model = chet_1_base(device=device)
    
    black_model = Chet(ModelConfig(
        embed_dim=768,
        n_heads=12,
        n_layers=12,
    ))
    black_model.load_state_dict(torch.load(".data/model_best.pth", map_location=device))
    # --------------------------

    N_GAMES = 5000
    TEMPERATURE = 0.5

    print(f"Playing {N_GAMES} games (temperature={TEMPERATURE}, device={device})\n")
    results = play_match(
        white_model, black_model, N_GAMES,
        device=device, temperature=TEMPERATURE,
    )

    w, d, b = results["white"], results["draw"], results["black"]
    print(f"\nFinal: +{w} ={d} -{b}")
