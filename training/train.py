from .dataset import ChessDataset, load_dataset
import csv
import math
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from chet.model_v2 import Chet, ModelConfig


def train(
    model: Chet,
    train_dataset: ChessDataset,
    val_dataset: ChessDataset | None = None,
    *,
    batch_size: int = 512,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    warmup_steps: int | None = 100000,
    device: str = "cuda",
    num_workers: int = 4,
    log_file: str = "training_log.csv",
    val_interval_minutes: float = 60,
    compile_model: bool = True,
) -> None:
    torch.backends.cudnn.benchmark = True

    model = model.to(device)

    if compile_model:
        model = torch.compile(model)

    model.train()

    # Create data loaders
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=use_persistent,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=use_persistent,
        )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Mixed precision: bf16 on A100 (no GradScaler needed for bf16)
    autocast_ctx = torch.amp.autocast(device, dtype=torch.bfloat16)

    # Setup learning rate scheduler: linear warmup + cosine decay
    total_steps = len(train_loader) * epochs
    if warmup_steps is None:
        warmup_steps = 0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps if warmup_steps > 0 else 1.0
        # Cosine decay from 1.0 to 0 over remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Validation helper
    last_val_loss = float("inf")

    def run_validation(global_step: int) -> None:
        nonlocal last_val_loss
        if val_loader is None:
            return

        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for boards, targets in val_loader:
                boards = boards.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast_ctx:
                    logits = model(boards)
                    loss = criterion(logits, targets)

                val_loss += loss.item()
                val_steps += 1

        val_loss /= val_steps
        print(f"\nValidation loss: {val_loss:.4f}")

        # Log validation loss
        examples_seen = global_step * batch_size
        now = time.monotonic()
        log_writer.writerow([now - start_time, examples_seen, "", f"{val_loss:.6f}", ""])
        log_f.flush()

        if val_loss < last_val_loss:
            last_val_loss = val_loss
            print(f"Saving model with validation loss {val_loss:.4f} as best model")
            # Unwrap torch.compile wrapper if present
            raw_model = getattr(model, "_orig_mod", model)
            torch.save(raw_model.state_dict(), "model_best.pth")

        model.train()

    # Training loop
    global_step = 0
    val_interval_secs = val_interval_minutes * 60
    train_log_interval_secs = 10
    last_val_time = time.monotonic()
    last_train_log_time = time.monotonic()
    train_log_loss_accum = 0.0
    train_log_count = 0

    log_f = open(log_file, "w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["time", "examples_seen", "train_loss", "val_loss", "lr"])
    log_f.flush()
    start_time = time.monotonic()

    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            i = 0
            for boards, targets in pbar:
                boards = boards.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Forward pass with mixed precision
                with autocast_ctx:
                    logits = model(boards)
                    loss = criterion(logits, targets)

                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Update running loss and progress bar
                running_loss += loss.item()
                i += 1
                global_step += 1
                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"train_loss": f"{running_loss / i:.4f}", "lr": f"{current_lr:.2e}"})

                # Accumulate training loss and log at intervals
                train_log_loss_accum += loss.item()
                train_log_count += 1
                now = time.monotonic()
                if now - last_train_log_time >= train_log_interval_secs:
                    avg_loss = train_log_loss_accum / train_log_count
                    examples_seen = global_step * batch_size
                    log_writer.writerow([now - start_time, examples_seen, f"{avg_loss:.6f}", "", f"{current_lr:.6e}"])
                    log_f.flush()
                    train_log_loss_accum = 0.0
                    train_log_count = 0
                    last_train_log_time = now

                # Time-based validation
                now = time.monotonic()
                if now - last_val_time >= val_interval_secs:
                    run_validation(global_step)
                    last_val_time = now

            # Also validate at epoch boundary
            run_validation(global_step)
            last_val_time = time.monotonic()
    finally:
        log_f.close()


DATA_DIR = "training/data/processed"

# configure as needed
MODEL_CONFIG = ModelConfig(
    embed_dim=768,
    n_heads=12,
    n_layers=12,
    dropout=0.1,
)

if __name__ == "__main__":
    train_dataset, val_dataset = load_dataset(DATA_DIR)

    model = Chet(MODEL_CONFIG)
    print(f"Model has {model.get_n_params():,} parameters")
    train(model, train_dataset, val_dataset)
