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

def _signpm1(z: jnp.ndarray) -> jnp.ndarray:
    """Return ±1 (treat 0 as +1)."""
    return jnp.where(z >= 0.0, 1.0, -1.0)

"""Define our new pair as done in flow_matching.py"""
def paired_detour_path_and_field(
    *,
    t: jnp.ndarray,
    theta_1: jnp.ndarray,
    key_noise: jax.Array,
    sigma_min: float = 1e-4,
    A: float = 8.0,
    a: jnp.ndarray | None = None
    ):
    d = theta_1.shape[0]

    # Choose a separating direction a for s(theta_1)=sign(a^T theta_1).
    if a is None:
        # default: use first coordinate
        a = jnp.zeros((d,)).at[0].set(1.0)

    s = _signpm1(jnp.dot(a, theta_1))  # in {±1}

    # Noise
    theta_0 = jax.random.normal(key_noise, shape=theta_1.shape)

    # OT-like sigma(t)
    sigma_t = 1.0 - (1.0 - sigma_min) * t
    sigma_prime = -(1.0 - sigma_min)

    # Bump b(t) with b(0)=b(1)=0
    b = A * jnp.sin(jnp.pi * t)
    db = A * jnp.pi * jnp.cos(jnp.pi * t)

    # e0
    e0 = jnp.zeros_like(theta_1).at[0].set(1.0)

    # ---- Paired path sample ----
    theta_t = t * theta_1 + sigma_t * theta_0 + b * s * e0

    # ---- Paired target field ----
    # Option 1 (recommended): conditional drift expressed via (t, theta_t, theta_1),
    # so it is truly u_t(theta | theta_1) (no explicit theta_0).

    # u_t(theta | theta_1) = theta_1 + (sigma'/sigma)*(theta - t*theta_1 - b*s*e0) + db*s*e0
    u_t = theta_1 + (sigma_prime / sigma_t) * (theta_t - t * theta_1 - b * s * e0) + db * s * e0

    return theta_t, u_t, theta_0, s

"""Same code from flow_matching.py with new pair"""
def fmpe_loss_bad_u(
    model: VectorFieldNetwork,
    state: eqx.nn.State,
    theta_1: jax.Array,
    x: jax.Array,
    key: jax.Array,
    sigma_min: float = 1e-4,
    alpha: float = 0.0,
    A: float = 8.0,
    a: jnp.ndarray | None = None,
) -> jax.Array:
    """Perte flow matching pour un seul exemple (à utiliser avec jax.vmap)."""
    key_t, key_noise, key_step = jax.random.split(key, 3)
    u = jax.random.uniform(key_t)
    t = u ** (1.0 + alpha)

    theta_t, u_t, _, _ = paired_detour_path_and_field(
        t=t, theta_1=theta_1, key_noise=key_noise,
        sigma_min=sigma_min, A=A, a=a    )

    v_t, state = model(t, theta_t, x, state, inference=False, key=key_step)
    return jnp.mean((v_t - u_t) ** 2), state

"""Same code from flow_matching.py with new pair"""
batch_fmpe_loss_bad_u = eqx.filter_vmap(
    fmpe_loss_bad_u,
    in_axes=(None, None, 0, 0, 0, None, None, None, None),
    out_axes=(0, None)
)

"""Same code from flow_matching.py with new pair"""
@eqx.filter_jit
def train_step_bad_u(
    model: VectorFieldNetwork,
    state: eqx.nn.State,
    opt_state: optax.OptState,
    theta_batch: jax.Array,
    x_batch: jax.Array,
    keys: jax.Array,
    optim: optax.GradientTransformation,
    sigma_min: float = 1e-4,
    alpha: float = 0.0,
    A: float = 8.0,
    a: jnp.ndarray | None = None,
) -> tuple[VectorFieldNetwork, eqx.nn.State, optax.OptState, jax.Array]:

    def loss_fn(model, state):
        losses, state = batch_fmpe_loss_bad_u(
            model, state, theta_batch, x_batch, keys, sigma_min, alpha, A, a
        )
        return jnp.mean(losses), state

    (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, state)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, loss


# Main
def run_benchmark_goodbad(
    task_name: str = "two_moons",
    num_simulations: int = 100_000,
    max_iterations: int = 100_000,
    num_eval_points: int = 6,
    eval_samples: int = 5_000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    sigma_min: float = 1e-4,
    alpha: float = 0.0,
    seeds: list = [42, 43, 44],  # 3 Seeds par défaut
    u_type: str = "good"
):
    print(f"=== Lancement du Benchmark sur {task_name} (Seeds: {seeds}) ===")
    task = sbibm.get_task(task_name)
    theta_dim = task.dim_parameters
    x_dim = task.dim_data
    reference = task.get_reference_posterior_samples(num_observation=1)
    x_obs = _to_jax(task.get_observation(num_observation=1)).squeeze(0)

    print("Génération des simulations...")
    prior = task.get_prior()
    simulator = task.get_simulator()
    thetas = prior(num_samples=num_simulations)
    xs = simulator(thetas)
    theta_jax, x_jax = _to_jax(thetas), _to_jax(xs)

    eval_steps = np.unique(np.logspace(1, np.log10(max_iterations), num=num_eval_points, dtype=int))

    norm_types = ["none"]
    tolerances = [1e-5]
    a = None
    A = 10

    # Dictionnaire imbriqué : results[norm][metric][seed] = [valeurs...]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    best_model = None
    best_state = None

    for norm in norm_types:
        print(f"\n" + "="*50)
        print(f" NORMALISATION : {norm.upper()}")
        print("="*50)

        for seed in seeds:
            print(f"\n---> Seed {seed} en cours...")
            key = jax.random.PRNGKey(seed)
            key_model, key_train, key_sample = jax.random.split(key, 3)

            model, state = eqx.nn.make_with_state(VectorFieldNetwork)(
                theta_dim, x_dim, [32, 128, 512, 128, 32], depth_per_block=2, key=key_model, norm_type=norm
            )
            optim = optax.adam(learning_rate)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

            step_losses = []
            start_train_time = time.time()

            for step in range(1, max_iterations + 1):
                key_train, key_batch = jax.random.split(key_train)
                idx = jax.random.randint(key_batch, (batch_size,), 0, num_simulations)
                keys_b = jax.random.split(key_batch, batch_size)

                if u_type == "good" :

                  model, state, opt_state, loss = train_step(
                      model, state, opt_state,
                      theta_jax[idx], x_jax[idx], keys_b,
                      optim, sigma_min, alpha,
                 )

                elif u_type == "bad" :

                  model, state, opt_state, loss = train_step_bad_u(
                      model, state, opt_state,
                      theta_jax[idx], x_jax[idx], keys_b,
                      optim, sigma_min, alpha,
                      A=A, a=a
                 )


                step_losses.append(float(loss))

                if step in eval_steps:
                    best_model = model
                    best_state = state
                    avg_loss = np.mean(step_losses[-100:]) if len(step_losses) > 100 else np.mean(step_losses)
                    print(f"  [Step {step:6d}] Loss: {avg_loss:.4f} | Total Time: {time.time() - start_train_time:.1f}s")

                    results[norm]["loss"][seed].append(avg_loss)

                    for tol in tolerances:
                        key_sample, subkey = jax.random.split(key_sample)
                        infer_path_type = "normal"

                        inf_start = time.time()

                        samples_jax, nfe = sample_posterior_with_stats(
                            model, state, x_obs, subkey, eval_samples, theta_dim, rtol=tol, atol=tol, path_type=infer_path_type
                        )
                        samples_jax.block_until_ready()
                        inf_time = time.time() - inf_start

                        samples_torch = _to_torch(samples_jax)
                        c2st_val = float(c2st(reference, samples_torch))

                        results[norm][f"c2st_{tol}"][seed].append(c2st_val)
                        results[norm][f"nfe_{tol}"][seed].append(float(nfe))
                        results[norm][f"inf_time_{tol}"][seed].append(inf_time)

    return results, eval_steps, best_model, best_state


# Viz
def _stack_seed_matrix(results_norm_metric_dict):
    """results_norm_metric_dict: dict[seed] -> list (len = n_eval_points)"""
    return np.array(list(results_norm_metric_dict.values()), dtype=float)

def plot_compare(results_good, results_bad, eval_steps, tol=1e-5, norm="none"):
    good_color = "red"
    bad_color  = "black"
    tol_key = f"{tol}"                 # "1e-05"
    c2st_key = f"c2st_{tol_key}"
    nfe_key  = f"nfe_{tol_key}"
    time_key = f"inf_time_{tol_key}"

    # FIGURE 1 : Loss + C2ST
    fig1, axs1 = plt.subplots(1, 2, figsize=(16, 6))

    # Loss (good)
    loss_good = _stack_seed_matrix(results_good[norm]["loss"])
    lg_mean, lg_min, lg_max = loss_good.mean(0), loss_good.min(0), loss_good.max(0)
    axs1[0].plot(eval_steps, lg_mean, color=good_color, marker='o', label="GOOD")
    axs1[0].fill_between(eval_steps, lg_min, lg_max, color=good_color, alpha=0.2)

    #Loss (bad)
    loss_bad = _stack_seed_matrix(results_bad[norm]["loss"])
    lb_mean, lb_min, lb_max = loss_bad.mean(0), loss_bad.min(0), loss_bad.max(0)
    axs1[0].plot(eval_steps, lb_mean, color=bad_color, marker='o', label="BAD")
    axs1[0].fill_between(eval_steps, lb_min, lb_max, color=bad_color, alpha=0.12)

    axs1[0].set_xscale('log')
    axs1[0].set_yscale('log')
    axs1[0].set_title("KL vs Iterations (Good vs Bad)")
    axs1[0].set_xlabel("Iterations (Log)")
    axs1[0].set_ylabel("KL Divergence (Log)")
    axs1[0].legend()
    axs1[0].grid(True, which="both", ls="--", alpha=0.5)

    # C2ST (good)
    c2st_good = _stack_seed_matrix(results_good[norm][c2st_key])
    cg_mean, cg_min, cg_max = c2st_good.mean(0), c2st_good.min(0), c2st_good.max(0)
    axs1[1].plot(eval_steps, cg_mean, color=good_color, marker='s', label="GOOD")
    axs1[1].fill_between(eval_steps, cg_min, cg_max, color=good_color, alpha=0.2)

    # C2ST (bad)
    c2st_bad = _stack_seed_matrix(results_bad[norm][c2st_key])
    cb_mean, cb_min, cb_max = c2st_bad.mean(0), c2st_bad.min(0), c2st_bad.max(0)
    axs1[1].plot(eval_steps, cb_mean, color=bad_color, marker='s', label="BAD")
    axs1[1].fill_between(eval_steps, cb_min, cb_max, color=bad_color, alpha=0.12)

    axs1[1].set_xscale('log')
    axs1[1].set_title(f"C2ST vs Iterations | tol={tol:.0e} (0.5=Perfect)")
    axs1[1].set_xlabel("Iterations (Log)")
    axs1[1].set_ylabel("C2ST Score")
    axs1[1].legend()
    axs1[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # FIGURE 2 : NFE + Inference Time
    fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6))

    # NFE (good)
    nfe_good = _stack_seed_matrix(results_good[norm][nfe_key])
    ng_mean, ng_min, ng_max = nfe_good.mean(0), nfe_good.min(0), nfe_good.max(0)
    axs2[0].plot(eval_steps, ng_mean, color=good_color, marker='.', label="GOOD")
    axs2[0].fill_between(eval_steps, ng_min, ng_max, color=good_color, alpha=0.2)

    # NFE (bad)
    nfe_bad = _stack_seed_matrix(results_bad[norm][nfe_key])
    nb_mean, nb_min, nb_max = nfe_bad.mean(0), nfe_bad.min(0), nfe_bad.max(0)
    axs2[0].plot(eval_steps, nb_mean, color=bad_color, marker='.', label="BAD")
    axs2[0].fill_between(eval_steps, nb_min, nb_max, color=bad_color, alpha=0.12)

    axs2[0].set_xscale('log')
    axs2[0].set_yscale('log')
    axs2[0].set_title(f"[{norm.upper()}] ODE Stiffness (NFE) | tol={tol:.0e}")
    axs2[0].set_xlabel("Iterations (Log)")
    axs2[0].set_ylabel("NFE (Log)")
    axs2[0].legend()
    axs2[0].grid(True, which="both", ls="--", alpha=0.5)

    # Inference Time (good)
    time_good = _stack_seed_matrix(results_good[norm][time_key])
    tg_mean, tg_min, tg_max = time_good.mean(0), time_good.min(0), time_good.max(0)
    axs2[1].plot(eval_steps, tg_mean, color=good_color, marker='.', label="GOOD")
    axs2[1].fill_between(eval_steps, tg_min, tg_max, color=good_color, alpha=0.2)

    # Inference Time (bad)
    time_bad = _stack_seed_matrix(results_bad[norm][time_key])
    tb_mean, tb_min, tb_max = time_bad.mean(0), time_bad.min(0), time_bad.max(0)
    axs2[1].plot(eval_steps, tb_mean, color=bad_color, marker='.', label="BAD")
    axs2[1].fill_between(eval_steps, tb_min, tb_max, color=bad_color, alpha=0.12)

    axs2[1].set_xscale('log')
    axs2[1].set_title(f"[{norm.upper()}] Inference Time (Seconds) | tol={tol:.0e}")
    axs2[1].set_xlabel("Iterations (Log)")
    axs2[1].set_ylabel("Time (s)")
    axs2[1].legend()
    axs2[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def visualize_two_moons_samples(
    model_good, state_good,
    model_bad,  state_bad,
    task_name="two_moons",
    num_samples=10_000,
    seed=0,
    tol=1e-5,
    path_type_good="normal",
    path_type_bad="normal",
):
    task = sbibm.get_task(task_name)
    theta_dim = task.dim_parameters
    reference = task.get_reference_posterior_samples(num_observation=1).numpy()
    x_obs = _to_jax(task.get_observation(num_observation=1)).squeeze(0)

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)

    samples_good, nfe_g = sample_posterior_with_stats(
        model_good, state_good, x_obs, k1,
        num_samples=num_samples, theta_dim=theta_dim,
        rtol=tol, atol=tol, path_type=path_type_good
    )
    samples_good = np.array(samples_good)

    samples_bad, nfe_b = sample_posterior_with_stats(
        model_bad, state_bad, x_obs, k2,
        num_samples=num_samples, theta_dim=theta_dim,
        rtol=tol, atol=tol, path_type=path_type_bad
    )
    samples_bad = np.array(samples_bad)

    plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    plt.title("Reference posterior")
    plt.scatter(reference[:,0], reference[:,1], s=2)
    plt.axis("equal")

    plt.subplot(1,3,2)
    plt.title(f"GOOD samples (NFE≈{float(nfe_g):.0f})")
    plt.scatter(samples_good[:,0], samples_good[:,1], s=2)
    plt.axis("equal")

    plt.subplot(1,3,3)
    plt.title(f"BAD samples (NFE≈{float(nfe_b):.0f})")
    plt.scatter(samples_bad[:,0], samples_bad[:,1], s=2)
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

    return samples_good, samples_bad