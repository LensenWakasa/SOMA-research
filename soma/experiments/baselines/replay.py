"""
Baseline: Experience Replay.

Store memory_size=200 examples from each past task. During training on new task,
mix in memory_per_task=20 examples per old task. Simplest form: no generative
model, just stored examples.

Expected BT: ~-0.07

Usage:
    python -m soma.experiments.baselines.replay --dataset permuted_mnist --n_tasks 10 --memory_size 200
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


def run_replay_baseline(
    tasks: List[Dict],
    device: str = "cpu",
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    memory_size: int = 200,
    memory_per_task: int = 20,
    seed: int = 42,
) -> Dict:
    """Run experience replay baseline."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for replay baseline.")

    np.random.seed(seed)
    torch.manual_seed(seed)

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

    # Memory buffer: list of (x, y) arrays from past tasks
    memory: List[Dict] = []
    rng = np.random.RandomState(seed)

    for t, task in enumerate(tasks):
        print(f"[Replay] Task {t+1}/{len(tasks)}")

        x_train = task["x_train"].copy()
        y_train = task["y_train"].copy()

        # Mix in memory from past tasks
        if memory:
            replay_x = []
            replay_y = []
            for mem in memory:
                n = min(memory_per_task, len(mem["x"]))
                indices = rng.choice(len(mem["x"]), n, replace=False)
                replay_x.append(mem["x"][indices])
                replay_y.append(mem["y"][indices])

            replay_x = np.concatenate(replay_x)
            replay_y = np.concatenate(replay_y)
            x_train = np.concatenate([x_train, replay_x])
            y_train = np.concatenate([y_train, replay_y])

        x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=device)
        x_test = torch.tensor(task["x_test"], dtype=torch.float32, device=device)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        dataset = TensorDataset(x_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        model.train()
        for step in range(n_steps):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Store memory for this task
        n_mem = min(memory_size, len(task["x_train"]))
        mem_idx = rng.choice(len(task["x_train"]), n_mem, replace=False)
        memory.append({
            "x": task["x_train"][mem_idx],
            "y": task["y_train"][mem_idx],
        })

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(x_test).argmax(dim=1).cpu().numpy()
        acc = float((preds == task["y_test"]).mean())
        peak_accuracies[t] = acc

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
        "method": "replay",
        "backward_transfer": logs[-1]["bt"] if logs else 0.0,
        "final_accuracy": logs[-1]["accuracy"] if logs else 0.0,
        "memory_size": memory_size,
        "memory_per_task": memory_per_task,
        "tasks_completed": len(tasks),
    }

    print(f"\n[Replay] Final BT: {result['backward_transfer']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Experience replay baseline")
    parser.add_argument("--dataset", type=str, default="permuted_mnist")
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--memory_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from soma.experiments.run_permuted_mnist import generate_permuted_mnist
    tasks = generate_permuted_mnist(n_tasks=args.n_tasks, seed=args.seed)

    device = args.device
    if device == "cuda" and HAS_TORCH and not torch.cuda.is_available():
        device = "cpu"

    result = run_replay_baseline(tasks, device=device, memory_size=args.memory_size, seed=args.seed)

    output_dir = Path("outputs") / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "replay_results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
