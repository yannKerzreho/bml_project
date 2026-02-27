import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optax
from typing import Literal, List


# ---------------------------------------------------------------------------
# Réseau vecteur champ conditionné sur (t, θ, x)
# ---------------------------------------------------------------------------

NormType = Literal["none", "layer", "spectral"]

# 1. Define a single Block [(Linear -> Norm -> Activation) * depth]+ [Skip]
class FlowBlock(eqx.Module):
    linears: tuple
    norms: tuple
    dropouts: tuple
    skip_proj: eqx.nn.Linear
    depth: int = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, orig_in_dim: int, depth: int, norm_type: NormType, key: jax.Array, dropout: float):
        self.depth = depth
        keys = jax.random.split(key, 2*depth + 1)
        
        linears_list = []
        norms_list = []
        dropout_list = []

        curr_in = in_dim
        for i in range(depth):
            lin = eqx.nn.Linear(curr_in, out_dim, key=keys[2*i])
            if norm_type == "spectral":
                linears_list.append(eqx.nn.SpectralNorm(lin, weight_name="weight", key=keys[2*i+1]))
            else:
                linears_list.append(lin)

            if norm_type == "layer":
                norms_list.append(eqx.nn.LayerNorm(out_dim))
            else:
                norms_list.append(eqx.nn.Identity())
            
            dropout_list.append(eqx.nn.Dropout(dropout))
            curr_in = out_dim 

        self.linears = tuple(linears_list)
        self.norms = tuple(norms_list)
        self.dropouts = tuple(dropout_list)
        self.skip_proj = eqx.nn.Linear(orig_in_dim, out_dim, key=keys[-1])

    def __call__(self, h: jax.Array, x_orig: jax.Array, state: eqx.nn.State, inference: bool, key: jax.Array = None):
        h_main = h
        
        for i in range(self.depth):
            if isinstance(self.linears[i], eqx.nn.SpectralNorm):
                h_main, state = self.linears[i](h_main, state, inference=inference)
            else:
                h_main = self.linears[i](h_main)
                
            h_main = self.norms[i](h_main)
            
            # Il faut passer une clé unique pour chaque couche de dropout
            if not inference and key is not None:
                step_key = jax.random.fold_in(key, i)
                h_main = self.dropouts[i](h_main, key=step_key, inference=inference)
            else:
                h_main = self.dropouts[i](h_main, inference=inference)
                
            h_main = jax.nn.silu(h_main)

        h_skip = self.skip_proj(x_orig)
        return h_main + h_skip, state


class VectorFieldNetwork(eqx.Module):
    blocks: tuple
    out_linear: eqx.Module
    norm_type: str = eqx.field(static=True)

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        hidden_sizes: List[int],
        depth_per_block: int,
        key: jax.Array,
        norm_type: NormType = "none",
        dropout: float=0.2
    ):
        self.norm_type = norm_type
        num_blocks = len(hidden_sizes)
        keys = jax.random.split(key, num_blocks + 2)
        
        orig_in_dim = 1 + theta_dim + x_dim
        dims = [orig_in_dim] + hidden_sizes

        blocks_list = []
        for i in range(num_blocks):
            blocks_list.append(
                FlowBlock(
                    in_dim=dims[i], 
                    out_dim=dims[i+1], 
                    orig_in_dim=orig_in_dim,
                    depth=depth_per_block, # Pass the inner depth!
                    norm_type=norm_type, 
                    key=keys[i],
                    dropout=dropout
                )
            )
        self.blocks = tuple(blocks_list)

        out_lin = eqx.nn.Linear(dims[-1], theta_dim, key=keys[-2])
        if norm_type == "spectral":
            self.out_linear = eqx.nn.SpectralNorm(out_lin, weight_name="weight", key=keys[-1])
        else:
            self.out_linear = out_lin

    def __call__(
        self, t: jax.Array, theta: jax.Array, x: jax.Array, state: eqx.nn.State, inference: bool = False, key: jax.Array = None
    ) -> tuple[jax.Array, eqx.nn.State]:
        
        _x = jnp.concatenate([jnp.atleast_1d(t), theta, x])
        h = _x

        if not inference and key is not None:
            keys = jax.random.split(key, len(self.blocks))
        else:
            keys = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            h, state = block(h, _x, state, inference, keys[i])

        if isinstance(self.out_linear, eqx.nn.SpectralNorm):
            out, state = self.out_linear(h, state, inference=inference)
        else:
            out = self.out_linear(h)
            
        return out, state


# ---------------------------------------------------------------------------
# Perte FMPE — Eq. (7) du papier
# ---------------------------------------------------------------------------

def fmpe_loss(
    model: VectorFieldNetwork,
    state: eqx.nn.State,
    theta_1: jax.Array,
    x: jax.Array,
    key: jax.Array,
    sigma_min: float = 1e-4,
    alpha: float = 0.0,
) -> jax.Array:
    """Perte flow matching pour un seul exemple (à utiliser avec jax.vmap)."""
    key_t, key_noise, key_step = jax.random.split(key, 3)

    # Échantillonnage de t ~ p_alpha(t)  (Sec. 3.3)
    u = jax.random.uniform(key_t)
    t = u ** (1.0 + alpha)

    theta_0 = jax.random.normal(key_noise, shape=theta_1.shape)

    # Chemin optimal transport (Eq. 5)
    sigma_t = 1.0 - (1.0 - sigma_min) * t
    theta_t = t * theta_1 + sigma_t * theta_0

    # Champ cible conditionnel (Eq. 6)
    u_t = (theta_1 - (1.0 - sigma_min) * theta_t) / (1.0 - (1.0 - sigma_min) * t)

    # Champ prédit et perte MSE (Eq. 7)
    v_t, state = model(t, theta_t, x, state, inference=False, key=key_step)
    return jnp.mean((v_t - u_t) ** 2), state


batch_fmpe_loss = eqx.filter_vmap(
    fmpe_loss, 
    in_axes=(None, None, 0, 0, 0, None, None), 
    out_axes=(0, None) 
)

@eqx.filter_jit
def train_step(
    model: VectorFieldNetwork,
    state: eqx.nn.State,
    opt_state: optax.OptState,
    theta_batch: jax.Array,
    x_batch: jax.Array,
    keys: jax.Array,
    optim: optax.GradientTransformation,
    sigma_min: float = 1e-4,
    alpha: float = 0.0,
) -> tuple[VectorFieldNetwork, eqx.nn.State, optax.OptState, jax.Array]:
    """
    Une étape de gradient.
    eqx.filter_grad + has_aux=True :
      - différentie model (paramètres), pas state
      - state mis à jour retourné en auxiliaire
    """
    def loss_fn(model, state):
        losses, state = batch_fmpe_loss(
            model, state, theta_batch, x_batch, keys, sigma_min, alpha
        )
        return jnp.mean(losses), state

    (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, state)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, loss

# ---------------------------------------------------------------------------
# Echantillonnage posterior — intégration ODE de t=0 à t=1
# ---------------------------------------------------------------------------

@eqx.filter_jit
def sample_posterior(
    model: VectorFieldNetwork,
    state: eqx.nn.State,
    x_obs: jax.Array,
    key: jax.Array,
    theta_dim: int,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> jax.Array:
    theta_0 = jax.random.normal(key, shape=(theta_dim,))

    def vector_field_wrapper(t, y, args):
        # On active inference=True pour figer l'état !
        v_t, _ = model(t, y, x_obs, state, inference=True)
        return v_t

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field_wrapper),
        diffrax.Dopri5(),
        t0=0.0, t1=1.0, dt0=0.01,
        y0=theta_0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
    )
    return solution.ys[-1]


def sample_posterior_batch(
    model: VectorFieldNetwork,
    state: eqx.nn.State,
    x_obs: jax.Array,
    key: jax.Array,
    num_samples: int,
    theta_dim: int,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> jax.Array:
    """Génère `num_samples` échantillons en parallèle via vmap."""
    keys = jax.random.split(key, num_samples)
    return jax.vmap(
        lambda k: sample_posterior(model, state, x_obs, k, theta_dim, rtol, atol)
    )(keys)