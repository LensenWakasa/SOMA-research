# SOMA — Self-Organising Modular Architecture

**Wakasa Labs · Nairobi, Kenya · March 2026**

A continual learning framework for large language models that solves catastrophic forgetting by growing new LoRA adapter capacity when and only when existing capacity is genuinely exhausted.

## Paper 1 Hypothesis

> A continual learning system using RL-guided selective capacity growth, triggered by a three-signal necessity detector (loss plateau + subspace saturation + systematic failure clustering), achieves Backward Transfer > −0.05 on sequential task learning benchmarks, using fewer than n_tasks adapters, outperforming EWC, experience replay, and sequential fine-tuning.

## Quick Start

```bash
# Create environment
python -m venv soma_env
source soma_env/bin/activate  # Linux/Mac
soma_env\Scripts\activate     # Windows

# Install
pip install -e .

# Run unit tests (no GPU needed)
python -m pytest soma/tests/ -v

# Run Experiment 1 — Permuted MNIST
python -m soma.experiments.run_permuted_mnist --n_tasks 10 --device cuda
```

## Repository Structure

```
soma/
├── __init__.py
├── requirements.txt
├── core/                    # Algorithm implementation
│   ├── necessity.py         # N1+N2+N3 detectors
│   ├── grow.py              # RL policy, 4 actions, reward, KL gating
│   ├── router.py            # Prototype-based adapter routing
│   └── learn.py             # Outer training loop, metrics, logging
├── experiments/             # Runnable experiments
│   ├── run_permuted_mnist.py
│   ├── run_gsm8k_sequential.py
│   └── baselines/
│       ├── ewc.py
│       ├── replay.py
│       └── sequential.py
├── configs/                 # Hyperparameter configs
│   ├── default.json
│   ├── paper1_permuted_mnist.json
│   └── paper1_gsm8k.json
├── utils/                   # Shared utilities
│   ├── metrics.py
│   ├── logging.py
│   ├── checkpoint.py
│   └── visualise.py
├── tests/                   # Unit tests
│   ├── test_n1_plateau.py
│   ├── test_n2_subspace.py
│   ├── test_n3_clustering.py
│   ├── test_grow_reward.py
│   ├── test_router.py
│   └── test_learn_integration.py
├── notebooks/               # Kaggle/Colab notebooks
│   ├── 01_experiment1_permuted_mnist.ipynb
│   ├── 02_experiment2_gsm8k.ipynb
│   └── 03_ablation_necessity_signals.ipynb
└── paper/                   # LaTeX source
    ├── main.tex
    ├── figures/
    └── tables/
```

## Validation Criteria

| Component | PASS | FAIL |
|-----------|------|------|
| N1 unit test | >95% correct | Any misclassification on clear cases |
| N2 unit test | Within 5% of expected | Residual direction wrong |
| N3 unit test | All 4 synthetic cases correct | Any case wrong |
| Exp 1 — SOMA | BT > −0.05 AND K < 10 | BT ≤ −0.05 OR K ≥ 10 |
| Exp 2 — SOMA | BT > −0.05 | BT ≤ −0.05 |

## Compute Budget

All experiments run on **Kaggle free T4 GPU** (30h/week). Total estimated cost: **< $133**.

## License

MIT — Wakasa Labs 2026
