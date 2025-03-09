from .base_model import Chet, ModelConfig, _from_pretrained


class Chet34(Chet):
    def __init__(self):
        super().__init__(
            ModelConfig(embed_dim=480, n_heads=12, n_layers=12, dropout=0.1)
        )

    @classmethod
    def from_pretrained(cls, path: str, *, device: str = "cpu"):
        return _from_pretrained(cls, path, device=device)
