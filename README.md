# chet

Chet is a lightweight neural network chess engine trained on high-level human play.

### Architecture

Chet is a decoder-only transformer model based loosely on [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) that outputs move predictions directly without any explicit game tree search. Chet is smell, with a current model size of only 22M parameters.

<img src="./assets/architecture.png" alt="Architecture" width="500"/>

