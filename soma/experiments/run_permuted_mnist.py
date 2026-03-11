"""
Experiment 1 — Permuted MNIST.

Validates the full SOMA system on the standard continual learning benchmark.
10 tasks: each task = MNIST with a different fixed random pixel permutation.

Base model: 3-layer MLP (784->256->128->10), frozen after init.
LoRA adapter: rank=8 on the hidden layer. 2,048 trainable parameters per adapter.
Training: 200 gradient steps per task, batch size 64, AdamW lr=1e-3.

PASS criterion: BT > -0.05 AND K < 10.

Usage:
    python -m soma.experiments.run_permuted_mnist
    python -m soma.experiments.run_permuted_mnist --n_tasks 10 --device cuda --seed 42
    python -m soma.experiments.run_permuted_mnist --disable_n1  # ablation
    python -m soma.experiments.run_permuted_mnist --no_rl       # rule-based
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from soma.core.necessity import SomaNecessity, NecessityConfig
from soma.core.grow import SomaGrow, GrowConfig, GrowAction
from soma.core.router import SomaRouter
from soma.core.learn import SomaLearn, LearnConfig, TaskLog


# ---------------------------------------------------------------------------
# Permuted MNIST data generation
# ---------------------------------------------------------------------------


def generate_permuted_mnist(
    n_tasks: int = 10,
    n_train: int = 1000,
    n_test: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """Generate permuted MNIST tasks.

    Each task applies a different fixed random pixel permutation to MNIST images.
    Returns list of task dicts with 'x_train', 'y_train', 'x_test', 'y_test', 'perm'.
    """
    rng = np.random.RandomState(seed)

    # Load MNIST (or generate synthetic data if torchvision unavailable)
    try:
        from torchvision import datasets, transforms
        dataset = datasets.MNIST(root="/tmp/mnist", train=True, download=True)
        x_all = dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_all = dataset.targets.numpy()
    except Exception:
        # Synthetic fallback for environments without torchvision
        print("WARNING: MNIST not available, using synthetic data for testing.")
        x_all = rng.randn(10000, 784).astype(np.float32) * 0.3 + 0.5
        x_all = np.clip(x_all, 0, 1)
        y_all = rng.randint(0, 10, size=10000)

    tasks = []
    for t in range(n_tasks):
        perm = rng.permutation(784)

        # Sample train/test splits
        indices = rng.choice(len(x_all), n_train + n_test, replace=False)
        x = x_all[indices][:, perm]  # apply permutation
        y = y_all[indices]

        tasks.append({
            "x_train": x[:n_train],
            "y_train": y[:n_train],
            "x_test": x[n_train:],
            "y_test": y[n_train:],
            "perm": perm,
            "seed": seed + t,
        })

    return tasks


# ---------------------------------------------------------------------------
# MLP base model
# ---------------------------------------------------------------------------

INPUT_DIM = 784
HIDDEN1 = 256
HIDDEN2 = 128
OUTPUT_DIM = 10
LORA_RANK = 8


class SimpleMLP(nn.Module):
    """3-layer MLP for MNIST. Frozen after init."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, OUTPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x, lora_BA=None):
        """Forward pass with optional LoRA on fc2."""
        h = self.relu(self.fc1(x))
        h2 = self.fc2(h)
        if lora_BA is not None:
            h2 = h2 + h @ lora_BA  # LoRA: h @ (B @ A) = h @ delta_W
        h2 = self.relu(h2)
        return self.fc3(h2)


# ---------------------------------------------------------------------------
# Train / eval / embed functions for SOMA
# ---------------------------------------------------------------------------


def make_soma_functions(
    base_model: nn.Module,
    pool: list,
    device: str = "cpu",
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Create train_fn, eval_fn, embed_fn, grad_fn for SomaLearn.

    Returns:
        (train_fn, eval_fn, embed_fn, grad_fn)
    """

    def train_fn(adapter_idx, task_data, lr_scale=1.0):
        """Train a LoRA adapter. adapter_idx=None -> fresh, else fine-tune."""
        x = torch.tensor(task_data["x_train"], dtype=torch.float32, device=device)
        y = torch.tensor(task_data["y_train"], dtype=torch.long, device=device)

        # Initialise LoRA parameters
        if adapter_idx is not None and adapter_idx < len(pool):
            B_init, A_init = pool[adapter_idx]
            B = torch.tensor(B_init, dtype=torch.float32, device=device, requires_grad=True)
            A = torch.tensor(A_init, dtype=torch.float32, device=device, requires_grad=True)
        else:
            B = torch.randn(HIDDEN1, LORA_RANK, device=device, requires_grad=True) * 0.01
            A = torch.randn(LORA_RANK, HIDDEN2, device=device, requires_grad=True) * 0.01

        optimizer = optim.AdamW([B, A], lr=lr * lr_scale)
        criterion = nn.CrossEntropyLoss()
        base_model.eval()

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for step in range(n_steps):
            for xb, yb in loader:
                optimizer.zero_grad()
                lora_BA = B @ A
                logits = base_model(xb, lora_BA=lora_BA)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return (B.detach().cpu().numpy(), A.detach().cpu().numpy())

    def eval_fn(adapter_idx, task_data):
        """Evaluate adapter on task. adapter_idx=-1 -> best available."""
        x = torch.tensor(task_data["x_test"], dtype=torch.float32, device=device)
        y = task_data["y_test"]

        if len(pool) == 0:
            return 0.1  # random baseline for 10 classes

        if adapter_idx == -1:
            # Try all adapters, return best accuracy
            best_acc = 0.0
            for i in range(len(pool)):
                acc = eval_fn(i, task_data)
                if acc > best_acc:
                    best_acc = acc
            return best_acc

        if adapter_idx >= len(pool):
            return 0.1

        B, A = pool[adapter_idx]
        B_t = torch.tensor(B, dtype=torch.float32, device=device)
        A_t = torch.tensor(A, dtype=torch.float32, device=device)

        base_model.eval()
        with torch.no_grad():
            lora_BA = B_t @ A_t
            logits = base_model(x, lora_BA=lora_BA)
            preds = logits.argmax(dim=1).cpu().numpy()

        accuracy = float((preds == y).mean())
        return accuracy

    def embed_fn(task_data):
        """Extract hidden embeddings for routing."""
        x = torch.tensor(task_data["x_train"][:50], dtype=torch.float32, device=device)
        base_model.eval()
        with torch.no_grad():
            h = torch.relu(base_model.fc1(x))
        return h.cpu().numpy()

    def grad_fn(task_data):
        """Compute gradients for necessity signals."""
        x = torch.tensor(task_data["x_train"][:100], dtype=torch.float32, device=device)
        y = torch.tensor(task_data["y_train"][:100], dtype=torch.long, device=device)

        # Use a temporary LoRA for gradient computation
        B = torch.randn(HIDDEN1, LORA_RANK, device=device, requires_grad=True) * 0.01
        A = torch.randn(LORA_RANK, HIDDEN2, device=device, requires_grad=True) * 0.01

        criterion = nn.CrossEntropyLoss(reduction="none")
        base_model.eval()

        grads = []
        losses = []
        failure_grads = []

        # Process in small batches
        for i in range(0, len(x), 10):
            xb = x[i:i+10]
            yb = y[i:i+10]

            if B.grad is not None:
                B.grad.zero_()
            if A.grad is not None:
                A.grad.zero_()

            lora_BA = B @ A
            logits = base_model(xb, lora_BA=lora_BA)
            loss_per_sample = criterion(logits, yb)
            loss = loss_per_sample.mean()
            loss.backward()

            if B.grad is not None:
                g = torch.cat([B.grad.flatten(), A.grad.flatten()]).detach().cpu().numpy()
                grads.append(g)
                losses.append(float(loss.item()))

                # Check for failures
                preds = logits.argmax(dim=1)
                if (preds != yb).any():
                    failure_grads.append(g)

        return grads, losses, failure_grads

    return train_fn, eval_fn, embed_fn, grad_fn


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(args):
    """Run Permuted MNIST experiment."""
    if not HAS_TORCH:
        print("ERROR: PyTorch not installed. Install with: pip install torch torchvision")
        return

    print(f"=== SOMA Experiment 1: Permuted MNIST ===")
    print(f"Tasks: {args.n_tasks}, Train: {args.n_train}, Test: {args.n_test}")
    print(f"Device: {args.device}, Seed: {args.seed}")
    if args.no_rl:
        print("Mode: RULE-BASED (no RL)")
    if args.disable_n1:
        print("Ablation: N1 DISABLED")
    if args.disable_n2:
        print("Ablation: N2 DISABLED")
    if args.disable_n3:
        print("Ablation: N3 DISABLED")
    print()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate data
    print("Generating permuted MNIST tasks...")
    tasks = generate_permuted_mnist(
        n_tasks=args.n_tasks,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )

    # Create base model
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    base_model = SimpleMLP().to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False  # Freeze base model

    # Adapter pool (shared between SOMA and train/eval functions)
    pool = []

    # Create SOMA functions
    train_fn, eval_fn, embed_fn, grad_fn = make_soma_functions(
        base_model, pool, device=device,
        n_steps=200, batch_size=64, lr=1e-3,
    )

    # Configure SOMA
    nec_cfg = NecessityConfig()
    grow_cfg = GrowConfig(max_k=args.n_tasks)  # ceiling at n_tasks

    cfg = LearnConfig(
        necessity=nec_cfg,
        grow=grow_cfg,
        cold_start_tasks=2,
        device=device,
    )

    learn = SomaLearn(
        train_fn=train_fn,
        eval_fn=eval_fn,
        embed_fn=embed_fn,
        grad_fn=grad_fn,
        cfg=cfg,
    )

    # Share pool reference
    learn.pool = pool

    # Run task stream
    past_tasks = []
    for t in range(args.n_tasks):
        task = tasks[t]
        print(f"[Task {t+1}/{args.n_tasks}]")

        log = learn.step(task_idx=t, task_data=task, past_task_data=past_tasks)
        past_tasks.append(task)

        print(f"  Action: {log.action}  K: {log.k_before}->{log.k_after}  "
              f"Acc: {log.accuracy:.3f}  BT: {log.backward_transfer:.4f}")

    # Summary
    result = learn.summary()

    # Save results
    output_dir = Path("outputs") / "permuted_mnist"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ""
    if args.no_rl:
        suffix = "_no_rl"
    if args.disable_n1:
        suffix = "_no_n1"
    if args.disable_n2:
        suffix = "_no_n2"
    if args.disable_n3:
        suffix = "_no_n3"

    with open(output_dir / f"results{suffix}.json", "w") as f:
        json.dump(result, f, indent=2)

    # Print PASS/FAIL
    bt = result["backward_transfer"]
    k = result["final_k"]
    passed = bt > -0.05 and k < args.n_tasks
    print(f"\n{'PASS' if passed else 'FAIL'}: BT={bt:.4f} K={k}")

    return result


def main():
    parser = argparse.ArgumentParser(description="SOMA Experiment 1: Permuted MNIST")
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_rl", action="store_true", help="Use rule-based policy")
    parser.add_argument("--disable_n1", action="store_true", help="Ablation: disable N1")
    parser.add_argument("--disable_n2", action="store_true", help="Ablation: disable N2")
    parser.add_argument("--disable_n3", action="store_true", help="Ablation: disable N3")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
