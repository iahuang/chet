from typing import Optional
import chess

PIECE_CHARS: dict[chess.PieceType, tuple[str, str]] = {
    chess.PAWN: ("♟︎", "♙"),
    chess.KNIGHT: ("♞", "♘"),
    chess.BISHOP: ("♝", "♗"),
    chess.ROOK: ("♜", "♖"),
    chess.QUEEN: ("♛", "♕"),
    chess.KING: ("♚", "♔"),
}


def get_ansi_rgb_fg(r: int, g: int, b: int) -> str:
    """
    Return the ANSI escape sequence for the given RGB color.
    """

    return f"\u001b[38;2;{r};{g};{b}m"


def get_ansi_rgb_bg(r: int, g: int, b: int) -> str:
    """
    Return the ANSI escape sequence for the given RGB color.
    """

    return f"\u001b[48;2;{r};{g};{b}m"


COLOR_FG_WHITE = get_ansi_rgb_fg(255, 255, 255)
COLOR_FG_BLACK = get_ansi_rgb_fg(0, 0, 0)
COLOR_FG_MAGENTA = get_ansi_rgb_fg(255, 100, 255)
COLOR_BG_BLACK_SQUARE = get_ansi_rgb_bg(209, 139, 71)
COLOR_BG_WHITE_SQUARE = get_ansi_rgb_bg(255, 206, 158)
COLOR_BG_SQUARE_HIGHLIGHT = get_ansi_rgb_bg(227, 83, 43)

RESET = "\u001b[0m"


def _print_square(
    piece: Optional[chess.Piece], is_white_square: bool, use_rgb: bool, draw_last_move_bg: bool
) -> None:
    if use_rgb:
        # draw background color
        if is_white_square:
            print(COLOR_BG_WHITE_SQUARE, end="")
        else:
            print(COLOR_BG_BLACK_SQUARE, end="")

        # draw last move background color
        if draw_last_move_bg:
            print(COLOR_BG_SQUARE_HIGHLIGHT, end="")

        # draw piece
        if piece is not None:
            # draw foreground color
            if piece.color == chess.WHITE:
                print(COLOR_FG_WHITE, end="")
            else:
                print(COLOR_FG_BLACK, end="")

            print(
                PIECE_CHARS[piece.piece_type][0], end=""
            )  # always use the white piece character, since RGB colors are used
        else:
            print(" ", end="")

        print(" ", end="")

    else:
        if piece is not None:
            print(PIECE_CHARS[piece.piece_type][piece.color], end="")
        else:
            print(" ", end="")

        print(" ", end="")

    # reset
    if use_rgb:
        print(RESET, end="")


def print_board(
    board: chess.Board,
    last_move: Optional[chess.Move] = None,
    *,
    show_move: bool = True,
    show_fen: bool = False,
    use_rgb: bool = False,
    description: Optional[str] = None,
) -> None:
    """
    Print the given board state to the terminal using Unicode chess piece characters.

    If `use_rgb` is `True`, the board will be colored using ANSI RGB escape sequences.
    Otherwise, the board will be printed without color.
    """

    top = "╭" + "─" * 20 + "╮"
    bottom = "╰" + "─" * 20 + "╯"

    if show_move:
        header = " White to move " if board.turn == chess.WHITE else " Black to move "

        # insert header text into top border
        header_start = (len(top) - len(header)) // 2
        top = top[:header_start] + header + top[header_start + len(header) :]

    description_lines = description.split("\n") if description is not None else []
    description_rank_start = 7

    if show_fen:
        description_rank_start -= 1
    
    description_rank_end = description_rank_start - len(description_lines)

    if description_rank_end < 0:
        raise ValueError("description is too long")

    print(top)

    for rank in range(7, -1, -1):
        print(f"│ {rank+1}", end="")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            is_white_square = (rank + file) % 2 == 1

            last_move_highlight = last_move is not None and square in (
                last_move.from_square,
                last_move.to_square,
            )
            
            _print_square(piece, is_white_square, use_rgb, last_move_highlight)
        print("  │", end="")

        if show_fen:
            if rank == 7:
                print(" FEN ", end="")
                if use_rgb:
                    print(COLOR_FG_MAGENTA, end="")
                print(board.fen(), end="")
                if use_rgb:
                    print(RESET, end="")
        if description_rank_start >= rank > description_rank_end:
            line = description_lines.pop(0)
            print(" " + line, end="")

        print()

    print("│  a b c d e f g h   │")
    print(bottom)
