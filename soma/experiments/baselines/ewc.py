"""
Baseline: EWC (Elastic Weight Consolidation).

Kirkpatrick et al. 2017. After each task, compute the diagonal Fisher information
matrix. Add a regularisation penalty to subsequent training:
    L_total = L_task + lambda * sum_i F_i * (theta_i - theta_i_*)^2

lambda=1000 is the standard value from the paper.

Expected BT: ~-0.09

Usage:
    python -m soma.experiments.baselines.ewc --dataset permuted_mnist --n_tasks 10 --lambda_ewc 1000
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


def compute_fisher(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    n_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    """Compute diagonal Fisher information matrix approximation."""
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    count = 0

    for xb, yb in data_loader:
        if count >= n_samples:
            break
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

        count += len(xb)

    for name in fisher:
        fisher[name] /= max(count, 1)

    return fisher


def run_ewc_baseline(
    tasks: List[Dict],
    device: str = "cpu",
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_ewc: float = 1000.0,
    seed: int = 42,
) -> Dict:
    """Run EWC baseline."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for EWC baseline.")

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

    # EWC state
    fisher_matrices: List[Dict[str, torch.Tensor]] = []
    optimal_params: List[Dict[str, torch.Tensor]] = []

    for t, task in enumerate(tasks):
        print(f"[EWC] Task {t+1}/{len(tasks)}")

        x_train = torch.tensor(task["x_train"], dtype=torch.float32, device=device)
        y_train = torch.tensor(task["y_train"], dtype=torch.long, device=device)
        x_test = torch.tensor(task["x_test"], dtype=torch.float32, device=device)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train with EWC penalty
        model.train()
        for step in range(n_steps):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)

                # EWC penalty from all past tasks
                ewc_loss = torch.tensor(0.0, device=device)
                for prev_fisher, prev_params in zip(fisher_matrices, optimal_params):
                    for name, param in model.named_parameters():
                        if name in prev_fisher:
                            ewc_loss += (prev_fisher[name] * (param - prev_params[name]) ** 2).sum()

                total_loss = loss + (lambda_ewc / 2.0) * ewc_loss
                total_loss.backward()
                optimizer.step()

        # Compute Fisher for this task
        fisher = compute_fisher(model, loader, device=device)
        fisher_matrices.append(fisher)
        optimal_params.append({name: param.clone().detach() for name, param in model.named_parameters()})

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
        "method": "ewc",
        "backward_transfer": logs[-1]["bt"] if logs else 0.0,
        "final_accuracy": logs[-1]["accuracy"] if logs else 0.0,
        "lambda": lambda_ewc,
        "tasks_completed": len(tasks),
    }

    print(f"\n[EWC] Final BT: {result['backward_transfer']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="EWC baseline")
    parser.add_argument("--dataset", type=str, default="permuted_mnist")
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--lambda_ewc", type=float, default=1000.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from soma.experiments.run_permuted_mnist import generate_permuted_mnist
    tasks = generate_permuted_mnist(n_tasks=args.n_tasks, seed=args.seed)

    device = args.device
    if device == "cuda" and HAS_TORCH and not torch.cuda.is_available():
        device = "cpu"

    result = run_ewc_baseline(tasks, device=device, lambda_ewc=args.lambda_ewc, seed=args.seed)

    output_dir = Path("outputs") / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ewc_results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
