"""
Baseline: Sequential Fine-tuning (no protection).

The no-protection baseline. Fine-tune the same model sequentially on each task.
No anti-forgetting mechanism. Establishes the worst-case ceiling.

Expected BT: ~-0.18

Usage:
    python -m soma.experiments.baselines.sequential --dataset permuted_mnist --n_tasks 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def run_sequential_baseline(
    tasks: List[Dict],
    device: str = "cpu",
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict:
    """Run sequential fine-tuning baseline.

    Same model, same training loop — just train sequentially with no protection.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for sequential baseline.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Simple MLP (same architecture as SOMA experiment)
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    peak_accuracies = {}
    logs = []

    for t, task in enumerate(tasks):
        print(f"[Sequential] Task {t+1}/{len(tasks)}")

        x_train = torch.tensor(task["x_train"], dtype=torch.float32, device=device)
        y_train = torch.tensor(task["y_train"], dtype=torch.long, device=device)
        x_test = torch.tensor(task["x_test"], dtype=torch.float32, device=device)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        model.train()
        for step in range(n_steps):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Evaluate on current task
        model.eval()
        with torch.no_grad():
            preds = model(x_test).argmax(dim=1).cpu().numpy()
        acc = float((preds == task["y_test"]).mean())
        peak_accuracies[t] = acc

        # Evaluate on all past tasks (BT)
        bt_diffs = []
        for i in range(t):
            x_old = torch.tensor(tasks[i]["x_test"], dtype=torch.float32, device=device)
            with torch.no_grad():
                preds_old = model(x_old).argmax(dim=1).cpu().numpy()
            acc_old = float((preds_old == tasks[i]["y_test"]).mean())
            bt_diffs.append(acc_old - peak_accuracies[i])

        bt = float(np.mean(bt_diffs)) if bt_diffs else 0.0
        print(f"  Acc: {acc:.3f}  BT: {bt:.4f}")

        logs.append({"task": t, "accuracy": acc, "bt": bt})

    result = {
        "method": "sequential",
        "backward_transfer": logs[-1]["bt"] if logs else 0.0,
        "final_accuracy": logs[-1]["accuracy"] if logs else 0.0,
        "tasks_completed": len(tasks),
    }

    print(f"\n[Sequential] Final BT: {result['backward_transfer']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Sequential fine-tuning baseline")
    parser.add_argument("--dataset", type=str, default="permuted_mnist")
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from soma.experiments.run_permuted_mnist import generate_permuted_mnist
    tasks = generate_permuted_mnist(n_tasks=args.n_tasks, seed=args.seed)

    device = args.device
    if device == "cuda" and HAS_TORCH and not torch.cuda.is_available():
        device = "cpu"

    result = run_sequential_baseline(tasks, device=device, seed=args.seed)

    output_dir = Path("outputs") / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sequential_results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
