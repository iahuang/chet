from .dataset import (
    ChessDataset,
    load_dataset,
    shuffle_dataset,
    split_dataset,
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from chet.model import Chet, ModelConfig
from chet.tokenizer import tokenize_board
import torch.nn.functional as F


def train(
    model: Chet,
    train_dataset: ChessDataset,
    val_dataset: ChessDataset | None = None,
    *,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    warmup_steps: int | None = None,
    device: str = "cuda",
) -> None:
    model = model.to(device)
    model.train()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Setup learning rate scheduler for warmup
    scheduler = None
    if warmup_steps is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0,
        )

    # Training loop
    global_step = 0
    last_val_loss = float("inf")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        # Create progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        i = 0
        for boards, targets in pbar:
            boards = boards.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(boards)

            # Compute loss (KL divergence between predicted and target distributions)
            loss = criterion(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate for warmup
            if scheduler is not None:
                scheduler.step()

            # Update running loss and progress bar
            running_loss += loss.item()
            i += 1
            pbar.set_postfix({"train_loss": f"{running_loss / i:.4f}"})

            global_step += 1

            # Print current learning rate during warmup
            if (
                warmup_steps is not None
                and global_step <= warmup_steps
                and global_step % 100 == 0
            ):
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Step {global_step}, LR: {current_lr:.6f}")

        # Validation phase
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for boards, targets in val_loader:
                    boards = boards.to(device)
                    targets = targets.to(device)

                    logits = model(boards)
                    loss = criterion(F.log_softmax(logits, dim=-1), targets)

                    val_loss += loss.item()
                    val_steps += 1

            val_loss /= val_steps
            print(f"Validation loss: {val_loss:.4f}")

        if val_loss is None or last_val_loss is None or val_loss < last_val_loss:
            last_val_loss = val_loss
            print(f"Saving model with validation loss {val_loss:.4f} as best model")
            torch.save(model.state_dict(), f"model_best.pth")


# fill this in
DATASET_PATH = "..."

# configure as needed
MODEL_CONFIG = ModelConfig(
    embed_dim=468,
    n_heads=12,
    n_layers=12,
    dropout=0.1,
)


if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH, tokenizer=tokenize_board)

    dataset = shuffle_dataset(dataset)
    train_dataset, val_dataset = split_dataset(dataset, 0.1)

    model = Chet(MODEL_CONFIG)
    train(model, train_dataset, val_dataset)
