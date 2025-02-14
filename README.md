# chet

Chet is a lightweight neural network chess engine trained on high-level human play.

## Architecture

Chet is a decoder-only transformer model based loosely on [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) that outputs move predictions directly without any explicit game tree search. Chet is smell, with a current largest model size of only 33M parameters.

Given a board state given as the set of pieces on each square as well as the side to move, the model outputs a probability distribution over all `4096 = 64 * 64` possible moves.

<img src="./assets/architecture.png" alt="Architecture" width="500"/>

## Dataset

Chet is trained on ~5 million chess positions taken from the [Lichess Elite Database](https://database.nikonoel.fr/), which was in turn derived from the [Lichess Open Database](https://database.lichess.org/). A proportion of the training data was also taken from the [Lichess Puzzles Dataset](https://database.lichess.org/#puzzles) in order to improve the model's explicit handling of tactical patterns.

## Pretrained Models

Weights for pretrained models are publically available but are not included in the repository due to their large size.

### Chet-33M

Model size: 32,535,893 parameters
Model file: [Google Drive](https://drive.google.com/file/d/1ypObrVRd_lXlVFABXb-o4u8koElxbiYE/view?usp=sharing)

```python
from chet.model import Chet, ModelConfig

config = ModelConfig(
    embed_dim=468,
    n_heads=12,
    n_layers=12,
    dropout=0.1,
)
model = Chet.from_pretrained("path/to/model.pt", config)
```

