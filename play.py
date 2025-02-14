from chet.model import Chet, ModelConfig
from chet.tokenizer import tokenize_board
from util.chessprint import print_board
import random
from typing import Literal, Callable
import chess
import torch


def play_game(
    model: Chet,
    tokenizer: Callable[[chess.Board], torch.Tensor],
    temperature: float = 1.0,
):
    """
    Play a game against the chess bot using the terminal interface.
    """

    # Initialize board
    board = chess.Board()
    last_move = None

    while not board.is_game_over():
        print_board(board, last_move=last_move, use_rgb=True)

        if board.turn == chess.WHITE:
            valid_move = False

            while not valid_move:
                try:
                    move_str = input(
                        '\nEnter your move (e.g. "e2e4" or simply "e4" when unambiguous) or "quit" to exit: '
                    )
                    if move_str.lower() == "quit":
                        return

                    if len(move_str) == 4 or len(move_str) == 5:
                        move = chess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            valid_move = True
                            last_move = move
                            board.push(move)
                        else:
                            print("Illegal move! Try again.")
                    elif len(move_str) == 2:
                        move_to = chess.SQUARE_NAMES.index(move_str)
                        selected_move = None

                        for move in board.legal_moves:
                            if move.to_square == move_to:
                                if selected_move is None:
                                    selected_move = move
                                else:
                                    print(
                                        f"Ambiguous move! {move} and {selected_move} are both legal moves."
                                    )
                                    selected_move = None
                                    break

                        if selected_move is not None:
                            valid_move = True
                            last_move = selected_move
                            board.push(selected_move)
                        else:
                            print("Illegal move! Try again.")
                except ValueError:
                    print("Invalid move format!")

        else:
            # Bot's turn (Black)
            print("\nBot is thinking...")

            # Get board tensor
            board_tensor = tokenizer(board).unsqueeze(0)

            # Get top moves from model
            top_moves = model.get_top_moves(
                board_tensor, board, n=5, temperature=temperature
            )

            # Print top moves being considered
            print(f"\nTop moves being considered (with temperature={temperature}):")
            for move, prob in top_moves:
                print(f"{move}: {prob:.1%}")

            # Sample a move from the top moves
            best_move = random.choices(
                top_moves, weights=[prob for _, prob in top_moves], k=1
            )[0][0]
            print(f"\nBot plays: {best_move}")

            last_move = best_move
            board.push(best_move)

    # Game over - print final state
    print_board(board, last_move=last_move, use_rgb=True)

    # Print game result
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"\nCheckmate! {winner} wins!")
    elif board.is_stalemate():
        print("\nStalemate! Game is drawn.")
    elif board.is_insufficient_material():
        print("\nDraw by insufficient material!")
    elif board.is_fifty_moves():
        print("\nDraw by fifty-move rule!")
    elif board.is_repetition():
        print("\nDraw by repetition!")
    else:
        print("\nGame Over!")


if __name__ == "__main__":
    config = ModelConfig(
        embed_dim=468,
        n_heads=12,
        n_layers=12,
        dropout=0.1,
    )
    path = "path/to/model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Chet.from_pretrained(path, config, device=device)

    try:
        play_game(
            model,
            tokenizer=lambda board: tokenize_board(board).to(device),
            temperature=0.5,
        )
    except KeyboardInterrupt:
        print("\nGame terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
