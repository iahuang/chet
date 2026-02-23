"""LoRA fine-tuning for Chet v2.

Decomposes the fused nn.MultiheadAttention QKV into explicit Q/K/V/O
projections, wraps target linear layers with low-rank adapters, and trains
only the adapter weights. Produces checkpoints compatible with the original
Chet model (fused QKV format).

Usage:
    python -m training.finetune_lora \
        --checkpoint model_best.pth \
        --data-dir training/data/ \
        --output-dir lora_output/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from chet.model_v2 import Chet, ModelConfig
from .dataset import load_dataset


# ---------------------------------------------------------------------------
# LoRA linear wrapper
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank adapter.

    output = base(x) + (x @ A^T) @ B^T * (alpha / rank)
    B is zero-initialized so the adapter is identity at init.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(rank, base.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B * self.scale

    def merged_weight(self) -> torch.Tensor:
        """Return base weight + LoRA delta (detached, for export)."""
        delta = (self.lora_B.T @ self.lora_A) * self.scale  # [out, in]
        return (self.base.weight.data + delta).detach()

    def merged_bias(self) -> torch.Tensor | None:
        return self.base.bias.data.detach() if self.base.bias is not None else None


# ---------------------------------------------------------------------------
# Decomposed multi-head attention (explicit Q/K/V/O projections)
# ---------------------------------------------------------------------------


class DecomposedAttention(nn.Module):
    """Self-attention with separate Q/K/V/O linear projections.

    Functionally equivalent to nn.MultiheadAttention(batch_first=True) but
    with individually addressable projection layers for LoRA injection.
    """

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention) -> DecomposedAttention:
        """Transfer weights from a fused nn.MultiheadAttention."""
        d = mha.embed_dim
        obj = cls(d, mha.num_heads)
        w = mha.in_proj_weight.data
        obj.q_proj.weight.data.copy_(w[:d])
        obj.k_proj.weight.data.copy_(w[d : 2 * d])
        obj.v_proj.weight.data.copy_(w[2 * d :])
        if mha.in_proj_bias is not None:
            b = mha.in_proj_bias.data
            obj.q_proj.bias.data.copy_(b[:d])
            obj.k_proj.bias.data.copy_(b[d : 2 * d])
            obj.v_proj.bias.data.copy_(b[2 * d :])
        obj.out_proj.weight.data.copy_(mha.out_proj.weight.data)
        if mha.out_proj.bias is not None:
            obj.out_proj.bias.data.copy_(mha.out_proj.bias.data)
        return obj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# LoRA-adapted transformer layer
# ---------------------------------------------------------------------------

ALL_TARGETS = {"q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"}
DEFAULT_TARGETS = ALL_TARGETS


class LoRATransformerLayer(nn.Module):
    """Transformer layer with decomposed attention and LoRA adapters.

    Takes ownership of the norms and FFN from the original layer and replaces
    the fused MHA with a DecomposedAttention. Target projections are wrapped
    with LoRALinear.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        alpha: float,
        targets: set[str] = DEFAULT_TARGETS,
    ) -> None:
        super().__init__()
        self.norm1 = original_layer.norm1
        self.norm2 = original_layer.norm2
        self.ffn = original_layer.ffn
        self.embed_dim = original_layer.embed_dim
        self.attn = DecomposedAttention.from_mha(original_layer.attn)

        attn_targets = targets & {"q_proj", "k_proj", "v_proj", "out_proj"}
        for name in attn_targets:
            base = getattr(self.attn, name)
            setattr(self.attn, name, LoRALinear(base, rank, alpha))

        ffn_targets = targets & {"w1", "w2", "w3"}
        for name in ffn_targets:
            base = getattr(self.ffn, name)
            setattr(self.ffn, name, LoRALinear(base, rank, alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Apply / export / merge
# ---------------------------------------------------------------------------


def apply_lora(
    model: Chet,
    rank: int = 16,
    alpha: float = 16.0,
    targets: set[str] = DEFAULT_TARGETS,
) -> Chet:
    """Freeze model and replace transformer layers with LoRA-adapted versions."""
    for p in model.parameters():
        p.requires_grad = False

    model.transformer_layers = nn.ModuleList(
        LoRATransformerLayer(layer, rank, alpha, targets)
        for layer in model.transformer_layers
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: rank={rank}, alpha={alpha}, targets={sorted(targets)}")
    print(f"LoRA: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)")
    return model


def get_lora_state_dict(model: Chet) -> dict[str, torch.Tensor]:
    """Extract only the trainable LoRA parameters."""
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_lora_weights(model: Chet, path: str, device: str = "cpu") -> None:
    """Load saved LoRA weights into a model that already has adapters applied."""
    lora_sd = torch.load(path, map_location=device, weights_only=True)
    own = dict(model.named_parameters())
    for key, value in lora_sd.items():
        if key not in own:
            raise KeyError(f"LoRA key {key!r} not found in model")
        own[key].data.copy_(value)


def _get_weight(module: nn.Module) -> torch.Tensor:
    if isinstance(module, LoRALinear):
        return module.merged_weight()
    return module.weight.data.detach()


def _get_bias(module: nn.Module) -> torch.Tensor | None:
    if isinstance(module, LoRALinear):
        return module.merged_bias()
    b = getattr(module, "bias", None)
    return b.data.detach() if b is not None else None


def export_merged_state_dict(model: Chet) -> dict[str, torch.Tensor]:
    """Merge LoRA deltas and return a state_dict compatible with vanilla Chet."""
    sd: dict[str, torch.Tensor] = {}

    # Board embedder + final norm (unchanged)
    for name, param in model.board_embedder.named_parameters():
        sd[f"board_embedder.{name}"] = param.data.detach()
    for name, buf in model.board_embedder.named_buffers():
        sd[f"board_embedder.{name}"] = buf.detach()
    sd["norm.weight"] = model.norm.weight.data.detach()

    # Move predictor (unchanged)
    for name, param in model.move_predictor.named_parameters():
        sd[f"move_predictor.{name}"] = param.data.detach()

    # Transformer layers: reconstruct fused QKV format
    for i, layer in enumerate(model.transformer_layers):
        p = f"transformer_layers.{i}."
        sd[f"{p}norm1.weight"] = layer.norm1.weight.data.detach()
        sd[f"{p}norm2.weight"] = layer.norm2.weight.data.detach()
        sd[f"{p}embed_dim"] = torch.tensor(layer.embed_dim)  # stored as buffer if needed

        attn = layer.attn
        q_w = _get_weight(attn.q_proj)
        k_w = _get_weight(attn.k_proj)
        v_w = _get_weight(attn.v_proj)
        sd[f"{p}attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = _get_bias(attn.q_proj)
        if q_b is not None:
            k_b = _get_bias(attn.k_proj)
            v_b = _get_bias(attn.v_proj)
            sd[f"{p}attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        sd[f"{p}attn.out_proj.weight"] = _get_weight(attn.out_proj)
        out_b = _get_bias(attn.out_proj)
        if out_b is not None:
            sd[f"{p}attn.out_proj.bias"] = out_b

        ffn = layer.ffn
        for fname in ("w1", "w2", "w3"):
            proj = getattr(ffn, fname)
            sd[f"{p}ffn.{fname}.weight"] = _get_weight(proj)
            fb = _get_bias(proj)
            if fb is not None:
                sd[f"{p}ffn.{fname}.bias"] = fb

    return sd


def merge_and_save(model: Chet, config: ModelConfig, save_path: str) -> None:
    """Merge LoRA weights into base model and save a vanilla-compatible checkpoint."""
    merged_sd = export_merged_state_dict(model)

    clean = Chet(config)
    # Filter to only keys present in the clean model (skip embed_dim tensor etc.)
    clean_keys = set(clean.state_dict().keys())
    filtered = {k: v for k, v in merged_sd.items() if k in clean_keys}
    clean.load_state_dict(filtered)
    torch.save(clean.state_dict(), save_path)
    print(f"Merged model saved to {save_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def finetune(
    model: Chet,
    config: ModelConfig,
    train_dataset,
    val_dataset=None,
    *,
    batch_size: int = 512,
    learning_rate: float = 2e-4,
    min_lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    max_steps: int = 50_000,
    device: str = "cuda",
    num_workers: int = 4,
    log_file: str = "finetune_log.csv",
    val_interval_minutes: float = 15,
    output_dir: str = "lora_output",
    compile_model: bool = True,
) -> None:
    """Fine-tune a LoRA-adapted Chet model.

    Only adapter parameters are updated. Cosine LR schedule decays to min_lr
    over max_steps, then training stops. Best-validation LoRA and merged
    checkpoints are saved automatically.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    model = model.to(device)
    if compile_model:
        model = torch.compile(model)
    model.train()

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )
    criterion = nn.CrossEntropyLoss()
    autocast_ctx = torch.amp.autocast(device, dtype=torch.bfloat16)

    decay_steps = max(max_steps - warmup_steps, 1)
    min_lr_ratio = min_lr / learning_rate if learning_rate > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps if warmup_steps > 0 else 1.0
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")

    def run_validation(step: int) -> None:
        nonlocal best_val_loss
        if val_loader is None:
            return
        model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for boards, targets in val_loader:
                boards = boards.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast_ctx:
                    loss = criterion(model(boards), targets)
                total_loss += loss.item()
                n += 1
        val_loss = total_loss / n
        print(f"\n[step {step}] val_loss={val_loss:.4f}")

        now = time.monotonic()
        log_writer.writerow([now - t0, step * batch_size, "", f"{val_loss:.6f}", ""])
        log_f.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("  New best — saving checkpoints")
            raw = getattr(model, "_orig_mod", model)
            torch.save(get_lora_state_dict(raw), os.path.join(output_dir, "lora_best.pth"))
            merge_and_save(raw, config, os.path.join(output_dir, "merged_best.pth"))
        model.train()

    # Logging setup
    log_path = os.path.join(output_dir, log_file)
    log_f = open(log_path, "w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["time", "examples_seen", "train_loss", "val_loss", "lr"])
    log_f.flush()
    t0 = time.monotonic()

    global_step = 0
    epoch = 0
    val_interval_secs = val_interval_minutes * 60
    last_val_time = t0
    last_log_time = t0
    log_loss_accum = 0.0
    log_count = 0

    try:
        while global_step < max_steps:
            epoch += 1
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
            i = 0

            for boards, targets in pbar:
                if global_step >= max_steps:
                    break

                boards = boards.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast_ctx:
                    loss = criterion(model(boards), targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                i += 1
                global_step += 1
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{running_loss / i:.4f}",
                    lr=f"{lr:.2e}",
                    step=f"{global_step}/{max_steps}",
                )

                log_loss_accum += loss.item()
                log_count += 1
                now = time.monotonic()
                if now - last_log_time >= 10:
                    avg = log_loss_accum / log_count
                    log_writer.writerow([
                        now - t0, global_step * batch_size, f"{avg:.6f}", "", f"{lr:.6e}",
                    ])
                    log_f.flush()
                    log_loss_accum = 0.0
                    log_count = 0
                    last_log_time = now

                if now - last_val_time >= val_interval_secs:
                    run_validation(global_step)
                    last_val_time = time.monotonic()

        # Final validation + checkpoint
        run_validation(global_step)
        raw = getattr(model, "_orig_mod", model)
        torch.save(get_lora_state_dict(raw), os.path.join(output_dir, "lora_final.pth"))
        print(f"\nDone. {global_step} steps, checkpoints in {output_dir}/")
    finally:
        log_f.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODEL_CONFIG = ModelConfig(embed_dim=768, n_heads=12, n_layers=12)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Chet")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained .pth")
    parser.add_argument("--data-dir", default="training/d2/")
    parser.add_argument("--output-dir", default="lora_output/")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=sorted(DEFAULT_TARGETS),
        choices=sorted(ALL_TARGETS),
        help="Which projections to adapt (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--val-interval", type=float, default=15, help="Minutes between validations")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    targets = set(args.targets)

    print(f"Loading base model from {args.checkpoint}")
    model = Chet.from_pretrained(args.checkpoint, MODEL_CONFIG)
    model = apply_lora(model, rank=args.rank, alpha=args.alpha, targets=targets)

    train_dataset, val_dataset = load_dataset(args.data_dir)
    print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")

    finetune(
        model,
        MODEL_CONFIG,
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        val_interval_minutes=args.val_interval,
        output_dir=args.output_dir,
        compile_model=not args.no_compile,
    )


if __name__ == "__main__":
    main()
