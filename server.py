from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chet.model import Chet, ModelConfig
from chet.tokenizer import tokenize_board
import chess
import torch
from typing import List, Tuple
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

global_fen = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
config = ModelConfig(embed_dim=480, n_heads=12, n_layers=12, dropout=0.1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Chet.from_pretrained(
    ".data/chet_34a.pt", config, device=device  # Update with actual model path
)
model.eval()


# Define request/response models
class MoveRequest(BaseModel):
    fen: str
    n: int = 5


class Move(BaseModel):
    uci: str
    probability: float
    relative_probability: float


class MoveResponse(BaseModel):
    moves: List[Move]
    fen: str


@app.post("/top_moves", response_model=MoveResponse)
async def predict_moves(request: MoveRequest):
    try:
        # Create board from FEN
        board = chess.Board(request.fen)

        # Tokenize board
        board_tensor = tokenize_board(board).unsqueeze(0).to(device)

        # Get top moves
        top_moves = model.get_top_moves(board_tensor, board, n=request.n)

        # Format response
        total_prob = sum(prob for _, prob in top_moves)

        moves = [
            Move(
                uci=move.uci(), probability=prob, relative_probability=prob / total_prob
            )
            for move, prob in top_moves
        ]

        return MoveResponse(moves=moves, fen=board.fen())

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class SetFenRequest(BaseModel):
    fen: str


@app.post("/set_fen")
async def set_fen(request: SetFenRequest):
    global global_fen
    global_fen = request.fen
    return {"message": "FEN set"}


@app.get("/get_fen")
async def get_fen():
    global global_fen
    return {"fen": global_fen}
