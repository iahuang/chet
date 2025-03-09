from dataclasses import dataclass


@dataclass
class ModelConfig:
    embed_dim: int
    n_heads: int
    n_layers: int
    dropout: float

    def as_dict(self):
        return {
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)
