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
from flow_matching import *

def _to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.numpy())

def _to_torch(x: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.array(x))

def run_benchmark_vs_budget(
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
    seeds: list = [1, 2, 3]
):
    print(f"=== Lancement du Benchmark sur {task_name} ===")
    task = sbibm.get_task(task_name)
    theta_dim = task.dim_parameters
    x_dim = task.dim_data
    reference = task.get_reference_posterior_samples(num_observation=1)
    x_obs = _to_jax(task.get_observation(num_observation=1)).squeeze(0)
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Structure : results[norm][metric][budget] = [valeur_seed1, valeur_seed2, ...]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    norm_types = ["none", "layer", "spectral"]
    tolerances = [1e-3, 1e-5, 1e-7]
    
    for norm in norm_types:
        print(f"\n" + "="*60)
        print(f" NORMALISATION : {norm.upper()}")
        print("="*60)
        
        for budget in simulation_budgets:
            print(f"\n  >>> BUDGET DE SIMULATION : {budget} <<<")
            num_train = int(budget * train_fraction)

            for seed in seeds:
                print(f"      -> Seed {seed} en cours...")
                
                # --- GÉNÉRATION D'UN NOUVEAU DATASET ---
                # On fixe la seed PyTorch pour s'assurer que pour un couple (budget, seed) donné,
                # les 3 architectures voient exactement le même dataset, permettant une comparaison juste.
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                sim_batch_size = 1000
                thetas_list, xs_list = [], []
                for _ in range(int(np.ceil(budget / sim_batch_size))):
                    th = prior(num_samples=sim_batch_size)
                    xs_list.append(simulator(th))
                    thetas_list.append(th)
                    
                thetas_budget = torch.cat(thetas_list, dim=0)[:budget]
                xs_budget = torch.cat(xs_list, dim=0)[:budget]
                theta_jax_budget, x_jax_budget = _to_jax(thetas_budget), _to_jax(xs_budget)
                # ---------------------------------------

                key = jax.random.PRNGKey(seed)
                key_split, key_model, key_train, key_sample = jax.random.split(key, 4)

                # Train / Eval Split
                indices = jax.random.permutation(key_split, budget)
                train_idx = indices[:num_train]
                theta_train, x_train = theta_jax_budget[train_idx], x_jax_budget[train_idx]

                model, state = eqx.nn.make_with_state(VectorFieldNetwork)(
                    theta_dim, x_dim, [32, 128, 512, 128, 32], depth_per_block=2, key=key_model, norm_type=norm
                )
                optim = optax.adam(learning_rate)
                opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

                best_c2st_1e7 = float('inf')
                epochs_without_improvement = 0
                num_batches = int(np.ceil(num_train / batch_size))
                best_metrics = {}

                for epoch in range(1, max_epochs + 1):
                    key_train, key_shuffle = jax.random.split(key_train)
                    shuffle_idx = jax.random.permutation(key_shuffle, num_train)
                    
                    # --- Training Loop ---
                    for b in range(num_batches):
                        batch_indices = shuffle_idx[b*batch_size : (b+1)*batch_size]
                        key_train, key_batch = jax.random.split(key_train)
                        keys_b = jax.random.split(key_batch, len(batch_indices))
                        
                        model, state, opt_state, loss = train_step(
                            model, state, opt_state,
                            theta_train[batch_indices], x_train[batch_indices], keys_b,
                            optim, sigma_min, alpha,
                        )

                    # --- Evaluation & Early Stopping ---
                    if epoch % eval_freq == 0 or epoch == 1:
                        current_metrics = {}
                        current_c2st_1e7 = None

                        for tol in tolerances:
                            key_sample, subkey = jax.random.split(key_sample)
                            
                            inf_start = time.time()
                            samples_jax, nfe = sample_posterior_with_stats(
                                model, state, x_obs, subkey, eval_samples, theta_dim, rtol=tol, atol=tol
                            )
                            samples_jax.block_until_ready()
                            inf_time = time.time() - inf_start
                            
                            samples_torch = _to_torch(samples_jax)
                            c2st_val = float(c2st(reference, samples_torch))
                            
                            current_metrics[f"c2st_{tol}"] = c2st_val
                            current_metrics[f"nfe_{tol}"] = float(nfe)
                            current_metrics[f"inf_time_{tol}"] = inf_time
                            
                            if tol == 1e-7:
                                current_c2st_1e7 = c2st_val
                                
                        # On garde la MEILLEURE époque
                        if current_c2st_1e7 < best_c2st_1e7:
                            best_c2st_1e7 = current_c2st_1e7
                            epochs_without_improvement = 0
                            best_metrics = current_metrics.copy()
                            print(f"         [Epoch {epoch:3d}] Nouveau meilleur C2ST (1e-7) : {best_c2st_1e7:.3f}")
                        else:
                            epochs_without_improvement += 1
                            if epochs_without_improvement >= patience:
                                print(f"         [!] Early stopping à l'époque {epoch}. Meilleur C2ST retenu : {best_c2st_1e7:.3f}")
                                break
                
                # Ajout des métriques de la meilleure époque aux résultats
                for metric_name, val in best_metrics.items():
                    results[norm][metric_name][budget].append(val)
                        
    return results, simulation_budgets


def plot_final_benchmark(results, simulation_budgets):
    norm_colors = {"none": "red", "layer": "blue", "spectral": "green"}
    tol_colors = {1e-3: "orange", 1e-5: "purple", 1e-7: "brown"}
    
    fig, axs = plt.subplots(3, 3, figsize=(24, 16))
    norm_types = ["none", "layer", "spectral"]
    tolerances = [1e-3, 1e-5, 1e-7]

    for i, norm in enumerate(norm_types):
        for tol in tolerances:
            # On extrait les moyennes, min et max en fonction du budget
            means_c2st, mins_c2st, maxs_c2st = [], [], []
            means_nfe, mins_nfe, maxs_nfe = [], [], []
            means_inf, mins_inf, maxs_inf = [], [], []
            
            for budget in simulation_budgets:
                # C2ST
                vals_c2st = results[norm][f"c2st_{tol}"][budget]
                means_c2st.append(np.mean(vals_c2st))
                mins_c2st.append(np.min(vals_c2st))
                maxs_c2st.append(np.max(vals_c2st))
                
                # NFE
                vals_nfe = results[norm][f"nfe_{tol}"][budget]
                means_nfe.append(np.mean(vals_nfe))
                mins_nfe.append(np.min(vals_nfe))
                maxs_nfe.append(np.max(vals_nfe))
                
                # Inference
                vals_inf = results[norm][f"inf_time_{tol}"][budget]
                means_inf.append(np.mean(vals_inf))
                mins_inf.append(np.min(vals_inf))
                maxs_inf.append(np.max(vals_inf))
                
            # -- Colonne 0 : C2ST Score --
            axs[i, 0].plot(simulation_budgets, means_c2st, color=tol_colors[tol], label=f"tol={tol:.0e}", marker='o')
            axs[i, 0].fill_between(simulation_budgets, mins_c2st, maxs_c2st, color=tol_colors[tol], alpha=0.2)

            # -- Colonne 1 : NFE --
            axs[i, 1].plot(simulation_budgets, means_nfe, color=tol_colors[tol], label=f"tol={tol:.0e}", marker='o')
            axs[i, 1].fill_between(simulation_budgets, mins_nfe, maxs_nfe, color=tol_colors[tol], alpha=0.2)

            # -- Colonne 2 : Inference Time --
            axs[i, 2].plot(simulation_budgets, means_inf, color=tol_colors[tol], label=f"tol={tol:.0e}", marker='o')
            axs[i, 2].fill_between(simulation_budgets, mins_inf, maxs_inf, color=tol_colors[tol], alpha=0.2)

        # Formattage global
        axs[i, 0].set_xscale('log')
        axs[i, 0].set_title(f"[{norm.upper()}] C2ST Score (0.5 = Perfect)")
        axs[i, 0].set_xlabel("Number of Simulations")
        axs[i, 0].set_ylabel("C2ST")
        axs[i, 0].legend()
        axs[i, 0].grid(True, which="both", ls="--", alpha=0.5)

        axs[i, 1].set_xscale('log')
        axs[i, 1].set_title(f"[{norm.upper()}] ODE Stiffness (NFE)")
        axs[i, 1].set_xlabel("Number of Simulations")
        axs[i, 1].set_ylabel("NFE")
        axs[i, 1].legend()
        axs[i, 1].grid(True, which="both", ls="--", alpha=0.5)

        axs[i, 2].set_xscale('log')
        axs[i, 2].set_title(f"[{norm.upper()}] Inference Time (Seconds)")
        axs[i, 2].set_xlabel("Number of Simulations")
        axs[i, 2].set_ylabel("Time (s)")
        axs[i, 2].legend()
        axs[i, 2].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()