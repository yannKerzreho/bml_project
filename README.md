# Flow Matching for Posterior Estimation — Normalization Benchmark

Experiment from my Bayesian Machine Learning course (MVA, ENS Paris-Saclay).  
Implementing and extending [Wildberger et al., NeurIPS 2023](https://arxiv.org/abs/2305.17161).

## What this is

**FMPE** trains a neural vector field $v_\phi(t,\theta,x)$ so that integrating

$$\dot{\theta}_t = v_\phi(t, \theta_t, x), \quad t \in [0,1]$$

maps a Gaussian $\mathcal{N}(0,I)$ to the posterior $p(\theta|x)$.  Training uses a
conditional flow-matching loss over simulated $(\theta, x)$ pairs with a straight
optimal-transport interpolant.  At inference, the ODE is solved with an adaptive
Dopri5 solver — and the **number of function evaluations (NFE)** depends directly on
how smooth (Lipschitz) the learned vector field is.

## Experiments

We benchmark across **two axes** on two sbibm tasks (Two Moons, SLCP):

| Axis | Values |
|------|--------|
| Simulation budget | 1 k / 10 k / 100 k |
| Weight normalization | None / Layer Norm / Spectral Norm |

For each of the 9 (norm × budget) combinations we run **5 independent seeds** with
early stopping, and measure:

| Metric | Description |
|--------|-------------|
| **C2ST** | Classifier two-sample test against reference posterior — 0.5 = perfect |
| **NFE** | ODE solver function evaluations per posterior sample |
| **Inference time** | Wall-clock time to draw 5 k samples |
| **Lipschitz constant** | Empirical $\hat{L}$ of $v_\phi(t{=}0.5, \cdot, x_\text{obs})$ tracked over training |

### Why normalization matters

**Layer Norm** stabilises the gradient scale across depth — most useful when data
is scarce (1 k budget) and the optimisation landscape is rough.

**Spectral Norm** constrains $\sigma_\max(W) \leq 1$ per layer, giving a product
upper bound $L \leq \prod_\ell \|W_\ell\|_2$ on the global Lipschitz constant.  A
Lipschitz-bounded $v_\phi$ produces a smoother, less stiff ODE — the adaptive solver
reaches tight tolerances with fewer steps.  We verify this directly:

1. **Lipschitz tracking** — plot $\hat{L}$ over training epochs for all three norms
2. **NFE vs tolerance curve** — sweep tolerances $10^{-2} \to 10^{-7}$ and measure how fast NFE grows

## Notebook structure (`main.ipynb`)

| Section | Content |
|---------|---------|
| 1 — Setup | Model config (swappable small / medium / large for CPU / GPU) |
| 2 — Two Moons | Run full benchmark |
| 3 — SLCP | Run full benchmark (5-D complex posterior task) |
| 4 — Results | 3 × 3 grid: C2ST / NFE / inference time vs budget |
| 5 — Lipschitz dynamics | Training-time Lipschitz curves per norm |
| 6 — NFE vs tolerance | Direct test of the spectral-norm ODE-smoothness claim |
| 7 — Discussion | Findings and practical recommendations |

## Repository layout

```
flow_matching.py   # VectorFieldNetwork, OT-FM loss, train_step, ODE sampler
utils.py           # Benchmark runner, Lipschitz estimator, all plotting helpers
main.ipynb         # Full experiment with theory, runs, and analysis
requirements.txt
```

## Setup

Requires **Python 3.10** for JAX/Equinox/Diffrax compatibility.

```bash
pip install -r requirements.txt
```

Open `main.ipynb` and run all cells.  A **GPU is strongly recommended** — the full
benchmark (2 tasks × 3 norms × 3 budgets × 5 seeds × up to 100 epochs) is expensive.
Swap `HIDDEN_SIZES` in the setup cell to use a larger model on GPU:

```python
# Small  (CPU):  [32, 128, 512, 128, 32]
# Medium (GPU):  [128, 512, 1024, 512, 128]
# Large  (GPU):  [256, 1024, 2048, 2048, 1024, 256]
```

### Running on Google Colab

You can push results back from Colab with standard git — just configure your credentials once:

```bash
git config --global user.email "you@example.com"
git config --global user.name  "Your Name"
git remote set-url origin https://<token>@github.com/<user>/<repo>.git
```

## Stack

Built entirely in [JAX](https://github.com/google/jax):
- [Equinox](https://github.com/patrick-kidger/equinox) — neural networks as PyTrees, SpectralNorm, LayerNorm
- [Diffrax](https://github.com/patrick-kidger/diffrax) — Dopri5 ODE solver with PID step-size control
- [Optax](https://github.com/google-deepmind/optax) — Adam optimiser
- [sbibm](https://github.com/sbi-benchmark/sbibm) — benchmark tasks (Two Moons, SLCP) and C2ST metric
