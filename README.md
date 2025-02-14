# chet

Chet is a lightweight neural network chess engine trained on high-level human play.

## Architecture

Chet is a decoder-only transformer model based loosely on [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) that outputs move predictions directly without any explicit game tree search. Chet is smell, with a current model size of only 35M parameters.

Given a board state given as the set of pieces on each square as well as the side to move, the model outputs a probability distribution over all `4096 = 64 * 64` possible moves.

<img src="./assets/architecture.png" alt="Architecture" width="500"/>

## Dataset

Chet is trained on ~5 million chess positions taken from the [Lichess Elite Database](https://database.nikonoel.fr/), which was in turn derived from the [Lichess Open Database](https://database.lichess.org/). A proportion of the training data was also taken from the [Lichess Puzzles Dataset](https://database.lichess.org/#puzzles) in order to improve the model's explicit handling of tactical patterns.