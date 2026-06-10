"""
FMPE Normalization Benchmark — experiment runner
=================================================
Trains FMPE with three weight normalization strategies (none / layer / spectral)
across a range of simulation budgets, on one or more sbibm tasks.
Results are serialised to results/ as JSON and can be visualised with analysis.ipynb.

Typical usage
-------------
# Local run, small model (CPU):
    python main.py

# Colab T4, medium model:
    python main.py --hidden 128 512 1024 512 128 --batch_size 256

# Single task, quick smoke-test (1 seed, 1 budget, 10 epochs):
    python main.py --tasks two_moons --budgets 10000 --seeds 1 --max_epochs 10
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import jax
from tasks import get_task
from utils import (
    run_benchmark,
    sample_posterior_with_stats,
    _to_jax,
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_serialisable(obj):
    """Recursively convert defaultdict / numpy types to JSON-serialisable Python."""
    if isinstance(obj, (defaultdict, dict)):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(_to_serialisable(data), f, indent=2)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# NFE vs tolerance sweep
# ---------------------------------------------------------------------------

def compute_nfe_vs_tolerance(best_models, best_states, task_name, tolerances, key,
                              n_samples=2_000, chunk_size=500):
    """
    For each normalization type, measure mean NFE at each ODE tolerance
    using the best trained model (largest budget, last seed).
    """
    task      = get_task(task_name)
    theta_dim = task.dim_parameters
    x_obs     = _to_jax(task.get_observation(num_observation=1)).squeeze(0)

    nfe_by_norm = {}
    for norm, model in best_models.items():
        state = best_states[norm]
        nfes  = []
        print(f"    NFE sweep [{norm}]", end="", flush=True)
        for tol in tolerances:
            key, subkey = jax.random.split(key)
            _, nfe = sample_posterior_with_stats(
                model, state, x_obs, subkey, n_samples, theta_dim,
                rtol=tol, atol=tol, chunk_size=chunk_size,
            )
            nfes.append(float(nfe))
            print(".", end="", flush=True)
        print()
        nfe_by_norm[norm] = nfes

    return {"tolerances": [float(t) for t in tolerances], "nfe_by_norm": nfe_by_norm}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FMPE Normalization Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks",       nargs="+", default=["two_moons", "slcp"],
                        help="sbibm tasks to benchmark")
    parser.add_argument("--budgets",     nargs="+", type=int,
                        default=[1_000, 10_000, 100_000],
                        help="simulation budgets")
    parser.add_argument("--seeds",       nargs="+", type=int,
                        default=[1, 2, 3, 4, 5],
                        help="random seeds")
    parser.add_argument("--hidden",      nargs="+", type=int,
                        default=[32, 128, 512, 128, 32],
                        help="hidden layer sizes  (e.g. 128 512 1024 512 128 for GPU)")
    parser.add_argument("--max_epochs",  type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=64,
                        help="increase to 256–512 on GPU")
    parser.add_argument("--eval_freq",   type=int, default=3)
    parser.add_argument("--patience",    type=int, default=7)
    parser.add_argument("--eval_samples",type=int, default=2_000,
                        help="posterior samples per eval (rounded to multiple of --sample_chunk)")
    parser.add_argument("--sample_chunk", type=int, default=500,
                        help="vmap chunk size for ODE sampling — trades GPU memory for latency")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    # Device info
    print(f"JAX devices : {jax.devices()}")
    print(f"Tasks       : {args.tasks}")
    print(f"Budgets     : {args.budgets}")
    print(f"Seeds       : {args.seeds}")
    print(f"Hidden      : {args.hidden}")
    print(f"Batch size  : {args.batch_size}")

    # Save run config so analysis.ipynb can read it
    config = {
        "tasks":        args.tasks,
        "budgets":      args.budgets,
        "seeds":        args.seeds,
        "hidden_sizes": args.hidden,
        "max_epochs":   args.max_epochs,
        "batch_size":   args.batch_size,
        "eval_freq":    args.eval_freq,
        "patience":     args.patience,
        "eval_samples": args.eval_samples,
    }
    save_json(config, results_dir / "config.json")

    nfe_tolerances = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7]
    key = jax.random.PRNGKey(0)

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"  Task: {task.upper()}")
        print(f"{'='*60}")

        results, budgets, best_models, best_states, lip_histories = run_benchmark(
            task_name         = task,
            simulation_budgets= args.budgets,
            max_epochs        = args.max_epochs,
            eval_freq         = args.eval_freq,
            eval_samples      = args.eval_samples,
            batch_size        = args.batch_size,
            patience          = args.patience,
            seeds             = args.seeds,
            hidden_sizes      = args.hidden,
            sample_chunk_size = args.sample_chunk,
        )

        # Save training results immediately — before the NFE sweep so a crash
        # there never loses the benchmark data.
        save_json(results,       results_dir / f"{task}_metrics.json")
        save_json(lip_histories, results_dir / f"{task}_lipschitz.json")

        print(f"\n  Computing NFE vs tolerance for {task}...")
        key, subkey = jax.random.split(key)
        nfe_vs_tol = compute_nfe_vs_tolerance(
            best_models, best_states, task, nfe_tolerances, subkey,
            chunk_size=args.sample_chunk,
        )
        save_json(nfe_vs_tol,    results_dir / f"{task}_nfe_vs_tol.json")

    print(f"\nAll done. Results in {results_dir}/")
    print("Open analysis.ipynb to visualise.")


if __name__ == "__main__":
    main()
