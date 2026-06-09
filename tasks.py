"""
tasks.py — Self-contained Two Moons and SLCP task definitions.

Drops the sbibm dependency entirely.  Reference posteriors and observations
are downloaded from sbibm's GitHub on first call and cached in data/ so
subsequent runs are instant.

Interface is intentionally compatible with sbibm:
    task = get_task("two_moons")
    task.dim_parameters          # int
    task.dim_data                # int
    task.get_prior()             # () -> (N, D) torch.Tensor
    task.get_simulator()         # (N, D) -> (N, K) torch.Tensor
    task.get_observation(1)      # -> (1, K) torch.Tensor
    task.get_reference_posterior_samples(1)  # -> (10_000, D) torch.Tensor
"""

import bz2
import io
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Reference files are served from sbibm's public GitHub repo
_BASE = "https://raw.githubusercontent.com/sbi-benchmark/sbibm/main/sbibm/tasks"
_DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def _load_reference(task_name: str, filename: str) -> torch.Tensor:
    """Download once, cache as .npy, return torch.Tensor."""
    cache = _DATA / task_name / (filename + ".npy")
    if cache.exists():
        return torch.tensor(np.load(cache), dtype=torch.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE}/{task_name}/files/num_observation_1/{filename}"
    print(f"  Fetching {task_name}/{filename} ...", end="", flush=True)
    raw = _fetch(url)
    if filename.endswith(".bz2"):
        raw = bz2.decompress(raw)
    df  = pd.read_csv(io.BytesIO(raw))
    arr = df.values.astype(np.float32)
    np.save(cache, arr)
    print(" cached.")
    return torch.tensor(arr, dtype=torch.float32)


# ---------------------------------------------------------------------------
# C2ST  (identical algorithm to sbibm.metrics.c2st)
# ---------------------------------------------------------------------------

def c2st(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    seed: int = 42,
    n_folds: int = 5,
) -> torch.Tensor:
    """
    Classifier two-sample test.
    Trains a Random Forest to distinguish p from q.
    Returns accuracy ∈ [0.5, 1.0] — 0.5 means indistinguishable (perfect posterior).
    """
    X = np.concatenate([samples_p.numpy(), samples_q.numpy()])
    y = np.concatenate([np.zeros(len(samples_p)), np.ones(len(samples_q))])
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy").mean()
    return torch.tensor(acc, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Base task class
# ---------------------------------------------------------------------------

class _Task:
    dim_parameters: int
    dim_data: int
    _name: str

    def get_prior(self):
        raise NotImplementedError

    def get_simulator(self):
        raise NotImplementedError

    def get_observation(self, num_observation: int = 1) -> torch.Tensor:
        assert num_observation == 1, "Only observation 1 is supported."
        return _load_reference(self._name, "observation.csv")

    def get_reference_posterior_samples(self, num_observation: int = 1) -> torch.Tensor:
        assert num_observation == 1, "Only observation 1 is supported."
        return _load_reference(self._name, "reference_posterior_samples.csv.bz2")


# ---------------------------------------------------------------------------
# Two Moons
# ---------------------------------------------------------------------------

class TwoMoons(_Task):
    """
    Prior   : theta ~ Uniform(-1, 1)^2
    Simulator:
        a ~ Uniform(-π/2, π/2)
        r ~ Normal(0.1, 0.01)
        p = [cos(a)·r + 0.25,  sin(a)·r]
        x = p + [-|rot(theta)[0]|,  rot(theta)[1]]   (rotation by -π/4)

    Source: Cranmer et al. 2020 / sbibm github.com/sbi-benchmark/sbibm
    """
    dim_parameters = 2
    dim_data       = 2
    _name          = "two_moons"

    def get_prior(self):
        def prior(num_samples: int = 1) -> torch.Tensor:
            return torch.empty(num_samples, 2).uniform_(-1.0, 1.0)
        return prior

    def get_simulator(self):
        def simulator(parameters: torch.Tensor) -> torch.Tensor:
            N = parameters.shape[0]
            a = torch.empty(N, 1).uniform_(-math.pi / 2, math.pi / 2)
            r = torch.empty(N, 1).normal_(0.1, 0.01)
            p = torch.cat([torch.cos(a) * r + 0.25, torch.sin(a) * r], dim=1)

            # _map_fun: rotate theta by -π/4, subtract abs of first component
            ang = torch.tensor(-math.pi / 4.0)
            c, s = torch.cos(ang), torch.sin(ang)
            z0 = (c * parameters[:, 0] - s * parameters[:, 1]).unsqueeze(1)
            z1 = (s * parameters[:, 0] + c * parameters[:, 1]).unsqueeze(1)
            return p + torch.cat([-torch.abs(z0), z1], dim=1)

        return simulator


# ---------------------------------------------------------------------------
# SLCP  (Simple Likelihood, Complex Posterior)
# ---------------------------------------------------------------------------

class SLCP(_Task):
    """
    Prior   : theta ~ Uniform(-3, 3)^5
    Simulator:
        mean = theta[0:2]
        s1 = theta[2]^2,  s2 = theta[3]^2,  rho = tanh(theta[4])
        Sigma = [[s1^2, rho·s1·s2], [rho·s1·s2, s2^2]]  + eps·I
        x_i ~ MVN(mean, Sigma)  for i = 1..4   → flatten to 8D

    Source: Papamakarios et al. 2019 / sbibm github.com/sbi-benchmark/sbibm
    """
    dim_parameters = 5
    dim_data       = 8
    _name          = "slcp"

    def get_prior(self):
        def prior(num_samples: int = 1) -> torch.Tensor:
            return torch.empty(num_samples, 5).uniform_(-3.0, 3.0)
        return prior

    def get_simulator(self):
        def simulator(parameters: torch.Tensor) -> torch.Tensor:
            N   = parameters.shape[0]
            mean = parameters[:, :2]                       # (N, 2)
            s1   = parameters[:, 2] ** 2                  # (N,)
            s2   = parameters[:, 3] ** 2                  # (N,)
            rho  = torch.tanh(parameters[:, 4])            # (N,)
            eps  = 1e-6

            cov = torch.zeros(N, 2, 2)
            cov[:, 0, 0] = s1 ** 2 + eps
            cov[:, 0, 1] = rho * s1 * s2
            cov[:, 1, 0] = rho * s1 * s2
            cov[:, 1, 1] = s2 ** 2 + eps

            # 4 i.i.d. draws from MVN(mean, cov) per parameter vector
            dist = torch.distributions.MultivariateNormal(
                mean.unsqueeze(1).expand(N, 4, 2).float(),
                cov.unsqueeze(1).expand(N, 4, 2, 2).float(),
            )
            return dist.sample().reshape(N, 8)   # (N, 8)

        return simulator


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_task(name: str) -> _Task:
    tasks = {"two_moons": TwoMoons, "slcp": SLCP}
    if name not in tasks:
        raise ValueError(f"Unknown task '{name}'. Available: {list(tasks)}")
    return tasks[name]()
