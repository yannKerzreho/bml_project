import time
import jax
import equinox as eqx
import optax
import numpy as np
import torch
import matplotlib.pyplot as plt
import sbibm
from sbibm.metrics import c2st
from collections import defaultdict
from tqdm.auto import tqdm
from flow_matching import *


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.numpy())


def load_results(results_dir, task_name):
    """
    Load saved benchmark data from results/<task>_*.json.
    JSON keys are always strings; this restores int budget keys so the
    plot functions work identically whether data came from run_benchmark()
    or from disk.

    Returns
    -------
    metrics       : dict  metrics[norm][metric][budget_int] = [val per seed]
    lip_histories : dict  lip_histories[norm][budget_int][seed_int] = [val per epoch]
    nfe_vs_tol    : dict  {"tolerances": [...], "nfe_by_norm": {norm: [nfe ...]}}
    config        : dict  run configuration (budgets, seeds, hidden_sizes, ...)
    """
    import json
    from pathlib import Path
    d = Path(results_dir)

    with open(d / f"{task_name}_metrics.json")    as f: raw_metrics  = json.load(f)
    with open(d / f"{task_name}_lipschitz.json")  as f: raw_lip      = json.load(f)
    with open(d / f"{task_name}_nfe_vs_tol.json") as f: nfe_vs_tol   = json.load(f)
    with open(d / "config.json")                  as f: config        = json.load(f)

    # Restore int budget keys
    metrics = {
        norm: {
            metric: {int(b): vals for b, vals in bdict.items()}
            for metric, bdict in mdict.items()
        }
        for norm, mdict in raw_metrics.items()
    }

    lip_histories = {
        norm: {
            int(b): {int(s): vals for s, vals in sdict.items()}
            for b, sdict in bdict.items()
        }
        for norm, bdict in raw_lip.items()
    }

    return metrics, lip_histories, nfe_vs_tol, config

def _to_torch(x: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.array(x))


# ---------------------------------------------------------------------------
# Lipschitz estimation
#
# We estimate the empirical Lipschitz constant of v_φ(t, ·, x_obs) at a
# fixed t=0.5 (mid-flow) by computing the maximum slope ratio over random
# pairs of parameter vectors:
#
#   L̂ = max_i  ‖v(θ_a^i) - v(θ_b^i)‖ / ‖θ_a^i - θ_b^i‖
#
# Spectral norm constrains each weight matrix to σ_max(W) ≤ 1, giving a
# product upper bound on L. Without normalization L is unconstrained and
# typically grows throughout training.
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _lipschitz_jit(model, state, x_obs, theta_a, theta_b, t_val):
    def get_v(theta):
        v, _ = model(t_val, theta, x_obs, state, inference=True)
        return v
    va = jax.vmap(get_v)(theta_a)
    vb = jax.vmap(get_v)(theta_b)
    dv     = jnp.linalg.norm(va - vb, axis=-1)
    dtheta = jnp.linalg.norm(theta_a - theta_b, axis=-1)
    return jnp.max(dv / (dtheta + 1e-8))


def estimate_lipschitz(model, state, x_obs, theta_dim, key, n_pairs=512, t_val=0.5):
    """Empirical Lipschitz estimate of v_φ(t=t_val, ·, x_obs) over random pairs."""
    k1, k2 = jax.random.split(key)
    theta_a = jax.random.normal(k1, (n_pairs, theta_dim))
    theta_b = jax.random.normal(k2, (n_pairs, theta_dim))
    return float(_lipschitz_jit(model, state, x_obs, theta_a, theta_b, jnp.array(t_val)))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    task_name: str = "two_moons",
    simulation_budgets: list = [1_000, 10_000, 100_000],
    train_fraction: float = 0.95,
    max_epochs: int = 100,
    eval_freq: int = 5,
    eval_samples: int = 5_000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    sigma_min: float = 1e-4,
    alpha: float = 0.0,
    patience: int = 3,
    seeds: list = [1, 2, 3, 4, 5],
    hidden_sizes: list = [32, 128, 512, 128, 32],
):
    """
    Train FMPE for every (normalization, budget, seed) combination and
    record posterior quality and ODE efficiency at the best epoch.

    Normalization types benchmarked
    --------------------------------
    none     : baseline — no weight normalization
    layer    : LayerNorm after each linear layer (stabilises gradient scale)
    spectral : SpectralNorm on each linear  (Lipschitz bound → smoother ODE)

    Returns
    -------
    results        : nested dict  results[norm][metric][budget] = [val per seed]
                     metrics: c2st_{tol}, nfe_{tol}, inf_time_{tol}, lipschitz
    budgets        : list of budgets run
    best_models    : {norm: model}  best model at the largest budget, last seed
    best_states    : {norm: state}
    lip_histories  : {norm: {budget: {seed: [lip_val per eval epoch]}}}
    """
    print(f"=== Benchmark: {task_name}  |  arch={hidden_sizes} ===")
    task      = sbibm.get_task(task_name)
    theta_dim = task.dim_parameters
    x_dim     = task.dim_data
    reference = task.get_reference_posterior_samples(num_observation=1)
    x_obs     = _to_jax(task.get_observation(num_observation=1)).squeeze(0)
    prior     = task.get_prior()
    simulator = task.get_simulator()

    results       = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    lip_histories = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    best_models, best_states = {}, {}

    norm_types = ["none", "layer", "spectral"]
    tolerances = [1e-3, 1e-5, 1e-7]

    for norm in norm_types:
        print(f"\n{'='*60}\n  Normalization: {norm.upper()}\n{'='*60}")

        for budget in simulation_budgets:
            print(f"\n  Budget: {budget:,}")
            num_train = int(budget * train_fraction)

            for seed in seeds:
                # -- generate dataset (fixed per (budget, seed) so norms see same data) --
                torch.manual_seed(seed)
                np.random.seed(seed)
                sim_batch = 1_000
                thetas_list, xs_list = [], []
                for _ in range(int(np.ceil(budget / sim_batch))):
                    th = prior(num_samples=sim_batch)
                    xs_list.append(simulator(th))
                    thetas_list.append(th)

                thetas_budget = torch.cat(thetas_list)[:budget]
                xs_budget     = torch.cat(xs_list)[:budget]
                theta_jax, x_jax = _to_jax(thetas_budget), _to_jax(xs_budget)

                key = jax.random.PRNGKey(seed)
                key_split, key_model, key_train, key_sample, key_lip = jax.random.split(key, 5)

                indices   = jax.random.permutation(key_split, budget)
                train_idx = indices[:num_train]
                theta_train, x_train = theta_jax[train_idx], x_jax[train_idx]

                model, state = eqx.nn.make_with_state(VectorFieldNetwork)(
                    theta_dim, x_dim, hidden_sizes,
                    depth_per_block=2, key=key_model, norm_type=norm,
                )
                optim     = optax.adam(learning_rate)
                opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

                num_batches       = int(np.ceil(num_train / batch_size))
                best_c2st_tight   = float("inf")
                epochs_no_improve = 0
                best_metrics      = {}
                best_model_state  = (model, state)

                pbar = tqdm(
                    range(1, max_epochs + 1),
                    desc=f"  {norm} / budget={budget:,} / seed={seed}",
                    leave=False,
                )

                for epoch in pbar:
                    key_train, key_shuffle = jax.random.split(key_train)
                    shuffle_idx = jax.random.permutation(key_shuffle, num_train)

                    for b in range(num_batches):
                        batch_idx = shuffle_idx[b * batch_size: (b + 1) * batch_size]
                        key_train, key_batch = jax.random.split(key_train)
                        keys_b = jax.random.split(key_batch, len(batch_idx))
                        model, state, opt_state, _ = train_step(
                            model, state, opt_state,
                            theta_train[batch_idx], x_train[batch_idx], keys_b,
                            optim, sigma_min, alpha,
                        )

                    if epoch % eval_freq == 0 or epoch == 1:
                        current_metrics    = {}
                        current_c2st_tight = None

                        # Lipschitz estimate at mid-flow t=0.5
                        key_lip, subkey_lip = jax.random.split(key_lip)
                        lip_val = estimate_lipschitz(
                            model, state, x_obs, theta_dim, subkey_lip,
                        )
                        lip_histories[norm][budget][seed].append(lip_val)
                        current_metrics["lipschitz"] = lip_val

                        for tol in tolerances:
                            key_sample, subkey = jax.random.split(key_sample)
                            t0 = time.time()
                            samples_jax, nfe = sample_posterior_with_stats(
                                model, state, x_obs, subkey, eval_samples,
                                theta_dim, rtol=tol, atol=tol,
                            )
                            samples_jax.block_until_ready()
                            inf_time = time.time() - t0

                            c2st_val = float(c2st(reference, _to_torch(samples_jax)))
                            current_metrics[f"c2st_{tol}"]     = c2st_val
                            current_metrics[f"nfe_{tol}"]      = float(nfe)
                            current_metrics[f"inf_time_{tol}"] = inf_time

                            if tol == 1e-7:
                                current_c2st_tight = c2st_val

                        pbar.set_postfix({
                            "C2ST": f"{current_c2st_tight:.3f}",
                            "Lip":  f"{lip_val:.1f}",
                        })

                        if current_c2st_tight < best_c2st_tight:
                            best_c2st_tight   = current_c2st_tight
                            epochs_no_improve = 0
                            best_metrics      = current_metrics.copy()
                            best_model_state  = (model, state)
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= patience:
                                pbar.close()
                                break

                for metric, val in best_metrics.items():
                    results[norm][metric][budget].append(val)

                # Retain best model at (largest budget, last seed) per norm
                if budget == max(simulation_budgets) and seed == seeds[-1]:
                    best_models[norm], best_states[norm] = best_model_state

    return results, simulation_budgets, best_models, best_states, lip_histories


# ---------------------------------------------------------------------------
# Plot — benchmark grid (C2ST / NFE / inference time)
# ---------------------------------------------------------------------------

def plot_results(results, simulation_budgets, task_name=""):
    """3 × 3 grid: rows = normalization, columns = C2ST / NFE / inference time."""
    tol_styles = {1e-3: ("orange", "o"), 1e-5: ("purple", "s"), 1e-7: ("brown", "^")}
    norm_types  = ["none", "layer", "spectral"]
    tolerances  = [1e-3, 1e-5, 1e-7]
    norm_labels = {"none": "No norm", "layer": "Layer Norm", "spectral": "Spectral Norm"}

    fig, axs = plt.subplots(3, 3, figsize=(18, 14))

    for row, norm in enumerate(norm_types):
        for tol in tolerances:
            tol_color, marker = tol_styles[tol]
            c2st_m, c2st_lo, c2st_hi = [], [], []
            nfe_m,  nfe_lo,  nfe_hi  = [], [], []
            inf_m,  inf_lo,  inf_hi  = [], [], []

            for budget in simulation_budgets:
                vc = results[norm][f"c2st_{tol}"][budget]
                vn = results[norm][f"nfe_{tol}"][budget]
                vi = results[norm][f"inf_time_{tol}"][budget]
                c2st_m.append(np.mean(vc)); c2st_lo.append(np.min(vc)); c2st_hi.append(np.max(vc))
                nfe_m.append(np.mean(vn));  nfe_lo.append(np.min(vn));  nfe_hi.append(np.max(vn))
                inf_m.append(np.mean(vi));  inf_lo.append(np.min(vi));  inf_hi.append(np.max(vi))

            for ax, means, lo, hi in [
                (axs[row, 0], c2st_m, c2st_lo, c2st_hi),
                (axs[row, 1], nfe_m,  nfe_lo,  nfe_hi),
                (axs[row, 2], inf_m,  inf_lo,  inf_hi),
            ]:
                ax.plot(simulation_budgets, means, color=tol_color, marker=marker,
                        label=f"tol={tol:.0e}")
                ax.fill_between(simulation_budgets, lo, hi, color=tol_color, alpha=0.15)

        lbl = norm_labels[norm]
        axs[row, 0].set_title(f"[{lbl}]  C2ST  (0.5 = perfect)", fontsize=11)
        axs[row, 1].set_title(f"[{lbl}]  ODE steps (NFE)",        fontsize=11)
        axs[row, 2].set_title(f"[{lbl}]  Inference time (s)",     fontsize=11)
        axs[row, 0].axhline(0.5, color="gray", linestyle=":", linewidth=1.2)

        for col, ylabel in enumerate(["C2ST", "NFE", "Time (s)"]):
            axs[row, col].set_xscale("log")
            axs[row, col].set_xlabel("Simulation budget")
            axs[row, col].set_ylabel(ylabel)
            axs[row, col].legend(fontsize=8)
            axs[row, col].grid(True, which="both", linestyle="--", alpha=0.4)

    title = "FMPE — weight normalization × simulation budget"
    if task_name:
        title += f"   |   Task: {task_name}"
    plt.suptitle(title + "\n5 seeds  |  early stopping on C2ST (tol=1e-7)", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot — Lipschitz constant over training
# ---------------------------------------------------------------------------

def plot_lipschitz_curves(lip_histories, simulation_budgets, eval_freq=5):
    """
    Empirical Lipschitz constant of v_φ(t=0.5, ·, x_obs) over training epochs.
    One panel per budget; shading = min–max across seeds.
    """
    norm_colors = {"none": "#d62728", "layer": "#1f77b4", "spectral": "#2ca02c"}
    norm_labels = {"none": "No norm", "layer": "Layer Norm", "spectral": "Spectral Norm"}
    norm_types  = ["none", "layer", "spectral"]

    n_panels = len(simulation_budgets)
    fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axs = [axs]

    for col, budget in enumerate(simulation_budgets):
        ax = axs[col]
        for norm in norm_types:
            seed_curves = list(lip_histories[norm][budget].values())
            if not seed_curves:
                continue
            min_len = min(len(c) for c in seed_curves)
            mat     = np.array([c[:min_len] for c in seed_curves])
            epochs  = np.arange(1, min_len + 1) * eval_freq

            mean = mat.mean(0)
            lo, hi = mat.min(0), mat.max(0)
            ax.plot(epochs, mean, color=norm_colors[norm], label=norm_labels[norm], linewidth=2)
            ax.fill_between(epochs, lo, hi, color=norm_colors[norm], alpha=0.2)

        ax.set_title(f"Budget: {budget:,}")
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Empirical Lipschitz  L̂(t=0.5)")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_yscale("log")

    plt.suptitle(
        "Lipschitz constant of v_φ(t=0.5, ·, x_obs) over training\n"
        "Spectral norm keeps L̂ ≈ 1 by construction; un-normed networks grow freely.",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot — NFE vs ODE tolerance
# ---------------------------------------------------------------------------

def plot_nfe_vs_tolerance(
    best_models: dict,
    best_states: dict,
    task_name: str,
    tolerances: list = None,
    n_samples: int = 2_000,
    seed: int = 0,
):
    """
    Sweep ODE tolerances and measure mean NFE for each normalization type.
    A Lipschitz-bounded vector field produces a less stiff ODE: the solver
    reaches tight tolerances with fewer function evaluations.

    Parameters
    ----------
    best_models / best_states : output of run_benchmark — best model per norm
    task_name   : sbibm task (used to retrieve x_obs and theta_dim)
    tolerances  : list of (rtol, atol) values to sweep (same value used for both)
    n_samples   : number of posterior samples per tolerance level
    """
    if tolerances is None:
        tolerances = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7]

    task      = sbibm.get_task(task_name)
    theta_dim = task.dim_parameters
    x_obs     = _to_jax(task.get_observation(num_observation=1)).squeeze(0)

    norm_colors = {"none": "#d62728", "layer": "#1f77b4", "spectral": "#2ca02c"}
    norm_labels = {"none": "No norm", "layer": "Layer Norm", "spectral": "Spectral Norm"}

    fig, ax = plt.subplots(figsize=(8, 5))
    key = jax.random.PRNGKey(seed)

    for norm, model in best_models.items():
        state = best_states[norm]
        nfes  = []
        for tol in tqdm(tolerances, desc=f"  tol sweep [{norm}]", leave=False):
            key, subkey = jax.random.split(key)
            _, nfe = sample_posterior_with_stats(
                model, state, x_obs, subkey, n_samples, theta_dim, rtol=tol, atol=tol,
            )
            nfes.append(float(nfe))

        ax.plot(tolerances, nfes, color=norm_colors[norm], marker="o",
                label=norm_labels[norm], linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("ODE tolerance  (tighter →)")
    ax.set_ylabel("Mean NFE per sample")
    ax.set_title(
        f"ODE stiffness: NFE vs tolerance — {task_name}\n"
        f"(model at 100 k budget, last seed)"
    )
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot — NFE vs tolerance from saved JSON data  (used by analysis.ipynb)
# ---------------------------------------------------------------------------

def plot_nfe_vs_tolerance_from_data(nfe_vs_tol: dict, task_name: str = ""):
    """
    Same plot as plot_nfe_vs_tolerance but reads from the JSON saved by main.py
    instead of running live ODE solves.  No JAX required.

    Parameters
    ----------
    nfe_vs_tol : dict loaded from results/<task>_nfe_vs_tol.json
                 {"tolerances": [...], "nfe_by_norm": {"none": [...], ...}}
    """
    norm_colors = {"none": "#d62728", "layer": "#1f77b4", "spectral": "#2ca02c"}
    norm_labels = {"none": "No norm", "layer": "Layer Norm", "spectral": "Spectral Norm"}

    tolerances  = nfe_vs_tol["tolerances"]
    nfe_by_norm = nfe_vs_tol["nfe_by_norm"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for norm, nfes in nfe_by_norm.items():
        ax.plot(tolerances, nfes, color=norm_colors.get(norm, "gray"),
                marker="o", label=norm_labels.get(norm, norm), linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("ODE tolerance  (tighter →)")
    ax.set_ylabel("Mean NFE per sample")
    title = "ODE stiffness: NFE vs tolerance"
    if task_name:
        title += f" — {task_name}"
    ax.set_title(title + "\n(model at 100 k budget, last seed)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
