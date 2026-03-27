"""
Sample (a1, a2, c1, c2) from the isotropic spin prior conditional on chi_eff and q.

Algorithm (see draft/main.tex for derivation):

  Stage 1 (analytic):
    p(chi | iso, amax) = ln(amax / |chi|) / (2 * amax),  |chi| < amax

  Stage 2 (numerical inverse-CDF):
    p(chi1 | chi_eff, q, iso) proportional to p(chi1|iso) * p(chi2|iso)
    where chi2 = ((1+q)*chi_eff - chi1) / q.
    The CDF is built on a uniform grid and inverted via linear interpolation.
    The batched version (sample_conditional_spins_batch) vmaps this over an
    array of chi_eff values in a single JIT-compiled call.

  Stage 3 (analytic inverse-CDF):
    P(c < tau | chi) = ln(amax * |tau| / |chi|) / ln(amax / |chi|)
    Inversion: |c| = |chi|^(1-u) * amax^(u-1),  a = |chi|^u * amax^(1-u)
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gwpopulation_pipe.analytic_spin_prior import chi_effective_prior_from_isotropic_spins

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Stage 1: marginal density p(chi | iso, amax)
# ---------------------------------------------------------------------------

def p_chi_iso(chi, amax=1.0):
    """
    Marginal density of chi = a * cos(theta) under isotropic spin prior
    with a ~ U[0, amax], cos(theta) ~ U[-1, 1].

    p(chi | iso, amax) = ln(amax / |chi|) / (2 * amax)  for |chi| < amax.
    """
    abs_chi = jnp.abs(chi)
    safe_chi = jnp.where(abs_chi > 0, abs_chi, 1.0)
    return jnp.where(abs_chi < amax, jnp.log(amax / safe_chi) / (2.0 * amax), 0.0)


# ---------------------------------------------------------------------------
# Stage 2: sample chi1 given chi_eff and q
# ---------------------------------------------------------------------------

def _chi1_unnorm_density(chi1, chi_eff, q, amax):
    """p(chi1|iso) * p(chi2|iso), unnormalized, with chi2 implied by chi_eff."""
    chi2 = ((1.0 + q) * chi_eff - chi1) / q
    return p_chi_iso(chi1, amax) * p_chi_iso(chi2, amax)


def _sample_chi1_single(chi_eff, q, u, amax, n_grid):
    """
    Pure-JAX, vmappable: sample one chi1 value given chi_eff, q, and a
    uniform variate u in [0, 1].  Builds the CDF on a grid of size n_grid.
    """
    chi1_lo = jnp.maximum(-amax, (1.0 + q) * chi_eff - q * amax)
    chi1_hi = jnp.minimum(amax, (1.0 + q) * chi_eff + q * amax)
    chi1_grid = jnp.linspace(chi1_lo, chi1_hi, n_grid)

    dens = jax.vmap(lambda c1: _chi1_unnorm_density(c1, chi_eff, q, amax))(chi1_grid)

    # Trapezoidal CDF, normalised to [0, 1]
    dx = (chi1_hi - chi1_lo) / (n_grid - 1)
    w = jnp.ones(n_grid).at[0].set(0.5).at[-1].set(0.5)
    cdf = jnp.cumsum(dens * w) * dx
    cdf = cdf / cdf[-1]

    return jnp.interp(u, cdf, chi1_grid)


# ---------------------------------------------------------------------------
# Stage 3: sample (c, a) given chi
# ---------------------------------------------------------------------------

def sample_c_given_chi(chi, u, amax=1.0):
    """
    Sample c = cos(theta) given chi = a * c via analytic CDF inversion.

    CDF:       P(c < tau | chi) = ln(amax * |tau| / |chi|) / ln(amax / |chi|)
    Inversion: |c| = |chi|^(1-u) * amax^(u-1)
    """
    abs_c = jnp.abs(chi) ** (1.0 - u) * amax ** (u - 1.0)
    return jnp.sign(chi) * abs_c


# ---------------------------------------------------------------------------
# Batched sampler  (preferred — single JIT call over all chi_eff values)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_grid",))
def sample_conditional_spins_batch(chi_eff_arr, q, key, amax=1.0, n_grid=1000):
    """
    Sample (a1, a2, c1, c2) for an array of chi_eff values in one JIT call.

    All per-sample work (CDF grid construction, inversion, analytic spin draw)
    is vmapped over the batch dimension, so there is no Python-level loop.

    Parameters
    ----------
    chi_eff_arr : jax array, shape (N,)
    q           : float
    key         : jax.random.PRNGKey
    amax        : float  (default 1.0)
    n_grid      : int    CDF grid size (default 1000)

    Returns
    -------
    a1, a2, c1, c2 : jax arrays, shape (N,)
    """
    k1, k2, k3 = jax.random.split(key, 3)
    N = chi_eff_arr.shape[0]

    u_chi1 = jax.random.uniform(k1, shape=(N,))
    u_c1   = jax.random.uniform(k2, shape=(N,))
    u_c2   = jax.random.uniform(k3, shape=(N,))

    def sample_one(chi_eff, u1, u2, u3):
        chi1 = _sample_chi1_single(chi_eff, q, u1, amax, n_grid)
        chi2 = ((1.0 + q) * chi_eff - chi1) / q
        c1   = sample_c_given_chi(chi1, u2, amax)
        c2   = sample_c_given_chi(chi2, u3, amax)
        return chi1 / c1, chi2 / c2, c1, c2

    a1, a2, c1, c2 = jax.vmap(sample_one)(chi_eff_arr, u_chi1, u_c1, u_c2)
    return a1, a2, c1, c2


# ---------------------------------------------------------------------------
# Single-chi_eff sampler (kept for convenience / CLI use)
# ---------------------------------------------------------------------------

def sample_conditional_spins(chi_eff, q, key, n_samples, amax=1.0, n_grid=2000):
    """
    Sample (a1, a2, c1, c2) from p(a1, a2, c1, c2 | chi_eff, q, iso).

    For drawing many samples at different chi_eff values, prefer
    sample_conditional_spins_batch which avoids the Python loop.

    Parameters
    ----------
    chi_eff   : float
    q         : float
    key       : jax.random.PRNGKey
    n_samples : int
    amax      : float  (default 1.0)
    n_grid    : int    (default 2000)

    Returns
    -------
    a1, a2, c1, c2 : jax arrays of shape (n_samples,)
    """
    chi_eff_arr = jnp.full((n_samples,), chi_eff)
    return sample_conditional_spins_batch(chi_eff_arr, q, key, amax, n_grid)


# ---------------------------------------------------------------------------
# Verification / demo
# ---------------------------------------------------------------------------

def verify(a1, a2, c1, c2, chi_eff, q, amax):
    if hasattr(chi_eff, "__len__"):
        chi_eff_check = (a1 * c1 + q * a2 * c2) / (1.0 + q)
        max_err = float(jnp.max(jnp.abs(chi_eff_check - jnp.array(chi_eff))))
        print(f"  Max |chi_eff_recovered - chi_eff_target| : {max_err:.2e}")
    else:
        chi_eff_recovered = (a1 * c1 + q * a2 * c2) / (1.0 + q)
        print(f"  chi_eff target              : {chi_eff:.4f}")
        print(f"  chi_eff recovered (mean)    : {float(jnp.mean(chi_eff_recovered)):.6f}")
        print(f"  chi_eff recovered (std)     : {float(jnp.std(chi_eff_recovered)):.2e}")
    print(f"  a1  in [0, amax]            : {bool(jnp.all((a1 >= 0) & (a1 <= amax)))}")
    print(f"  a2  in [0, amax]            : {bool(jnp.all((a2 >= 0) & (a2 <= amax)))}")
    print(f"  c1  in [-1, 1]              : {bool(jnp.all(jnp.abs(c1) <= 1))}")
    print(f"  c2  in [-1, 1]              : {bool(jnp.all(jnp.abs(c2) <= 1))}")


if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser(description="Sample spins conditional on chi_eff.")
    parser.add_argument("--chi_eff",   type=float, default=0.2)
    parser.add_argument("--q",         type=float, default=0.8)
    parser.add_argument("--amax",      type=float, default=1.0)
    parser.add_argument("--n_samples", type=int,   default=10_000)
    parser.add_argument("--n_grid",    type=int,   default=1000)
    parser.add_argument("--seed",      type=int,   default=0)
    args = parser.parse_args()

    print(f"chi_eff={args.chi_eff}, q={args.q}, amax={args.amax}, n_samples={args.n_samples}")

    key = jax.random.PRNGKey(args.seed)
    chi_eff_arr = jnp.full((args.n_samples,), args.chi_eff)

    # Warm-up (triggers JIT compilation)
    _ = sample_conditional_spins_batch(chi_eff_arr[:2], args.q, key, args.amax, args.n_grid)

    t0 = time.perf_counter()
    a1, a2, c1, c2 = sample_conditional_spins_batch(
        chi_eff_arr, args.q, key, args.amax, args.n_grid
    )
    jax.block_until_ready((a1, a2, c1, c2))
    print(f"  Time (after JIT): {time.perf_counter() - t0:.3f}s")

    verify(a1, a2, c1, c2, args.chi_eff, args.q, args.amax)
