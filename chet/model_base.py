from chet.config import ModelConfig
import torch.nn as nn
import torch
import chess


class BaseChet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

    def get_n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_top_moves(
        self,
        board_tokens: torch.Tensor,
        board: chess.Board,
        n: int = 5,
        *,
        temperature: float = 1.0
    ) -> list[tuple[chess.Move, float]]:
        raise NotImplementedError

    def load_weights(self, path: str, map_location: str = "cpu") -> None:
        weights = torch.load(path, map_location=map_location)

        weights = {_legacy_weight_name_map(k): v for k, v in weights.items()}

        self.load_state_dict(weights)
        self.to(map_location)


def _legacy_weight_name_map(name: str) -> str:
    if name.startswith("transformer_layers"):
        return name.replace("transformer_layers", "transformer.transformer_layers")

    if name.startswith("board_embedder"):
        return name.replace("board_embedder", "transformer.board_embedder")

    if name.startswith("norm"):
        return name.replace("norm", "transformer.norm")

    return name
