# chet

Chet is a lightweight neural network chess engine trained on high-level human play.

### Architecture

Chet is a decoder-only transformer model based loosely on [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) that outputs a probability distribution over the set of possible moves given a board state.
