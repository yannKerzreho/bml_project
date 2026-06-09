# Flow Matching for Posterior Estimation — Normalization Benchmark

Experiment from my Bayesian Machine Learning course (MVA, ENS Paris-Saclay).  
Implementing and extending [Wildberger et al., NeurIPS 2023](https://arxiv.org/abs/2305.17161).

## What this is

**FMPE** trains a neural vector field $v_\phi(t,\theta,x)$ so that integrating

$$\dot{\theta}_t = v_\phi(t, \theta_t, x), \quad t \in [0,1]$$

maps a Gaussian $\mathcal{N}(0,I)$ to the posterior $p(\theta|x)$.  Training uses a
conditional flow-matching loss over simulated $(\theta, x)$ pairs with a straight
optimal-transport interpolant.  At inference, the ODE is solved with adaptive Dopri5,
and the **number of function evaluations (NFE)** depends directly on how Lipschitz-smooth
the vector field is.

## Experiment

We benchmark three weight normalization strategies × three simulation budgets × two tasks:

| Axis | Values |
|------|--------|
| Simulation budget | 1 k / 10 k / 100 k |
| Normalization | None / Layer Norm / Spectral Norm |
| Task | Two Moons (2D) / SLCP (5D) |

5 seeds per (norm × budget) combination, with early stopping on C2ST.

| Metric | Description |
|--------|-------------|
| **C2ST** | Classifier two-sample test — 0.5 = perfect |
| **NFE** | ODE solver function evaluations per sample |
| **Inference time** | Wall-clock time for 5 k samples |
| **Lipschitz $\hat{L}$** | Empirical Lipschitz of $v_\phi(t{=}0.5,\cdot,x)$ tracked over training |

**Why spectral norm matters:** bounding $\sigma_\max(W)\leq 1$ per layer gives a product
upper bound on the global Lipschitz constant → smoother ODE → fewer solver steps at inference.

## Quickstart

```bash
pip install -r requirements.txt

# Small model, both tasks (CPU):
python main.py

# Medium model, both tasks (GPU / Colab T4):
python main.py --hidden 128 512 1024 512 128 --batch_size 256

# Then open analysis.ipynb to visualise results
```

## Running on Google Colab

Three cells are all you need:

```python
# 1 — Clone & install
!git clone https://github.com/yannKerzreho/bml_project.git
%cd bml_project
!pip install -q equinox diffrax optax sbibm tqdm

# 2 — Run (medium model, GPU)
!python main.py --hidden 128 512 1024 512 128 --batch_size 256

# 3 — Push results
!git config user.email "kzr.yann@gmail.com"
!git config user.name "yannKerzreho"
!git remote set-url origin https://YOUR_TOKEN@github.com/yannKerzreho/bml_project.git
!git add results/
!git commit -m "add results from colab T4"
!git push
```

Generate a token at `github.com → Settings → Developer settings → Personal access tokens` (needs `repo` scope).

## Repository layout

```
main.py          # Experiment runner — trains all models, saves to results/
analysis.ipynb   # Visualisation — loads results/ and plots all figures
flow_matching.py # VectorFieldNetwork, OT-FM loss, train_step, ODE sampler
utils.py         # run_benchmark, Lipschitz estimator, all plot helpers, load_results
results/         # JSON output from main.py (committed after Colab run)
requirements.txt
```

## CLI options

```
python main.py --help

--tasks        sbibm tasks to run (default: two_moons slcp)
--budgets      simulation budgets (default: 1000 10000 100000)
--seeds        random seeds (default: 1 2 3 4 5)
--hidden       hidden layer sizes (default: 32 128 512 128 32)
               GPU recommended: 128 512 1024 512 128
--batch_size   default 64; use 256-512 on GPU
--max_epochs   default 100
```

## Stack

[JAX](https://github.com/google/jax) ·
[Equinox](https://github.com/patrick-kidger/equinox) ·
[Diffrax](https://github.com/patrick-kidger/diffrax) (Dopri5 + PID) ·
[Optax](https://github.com/google-deepmind/optax) ·
[sbibm](https://github.com/sbi-benchmark/sbibm)
