"""1D Reaction-Diffusion solver — self-contained, no PDEBench import.

Equation (Fisher-KPP form, PDEBench convention):
    u_t = nu * u_xx + rho * u (1 - u)
on x in [0, 1] with periodic BCs. Initial conditions are PDEBench's
``init_multi`` (random multi-mode sinusoid) normalized to [0, 1].

Numerics: Strang splitting per inner step,
    half-step reaction (PDEBench's piecewise-exact solution)
    full-step diffusion (FFT, exact for the linear sub-problem)
    half-step reaction
This is unconditionally stable in dt and removes the explicit-FD CFL cap
``dt < dx^2 / (2 nu)``.
"""

from __future__ import annotations

from functools import partial
from math import ceil

import jax
import jax.numpy as jnp
from jax import lax, nn, random, vmap

# ---------------------------------------------------------------------------
# Domain & time params (matches the 1D Burgers production data: nx=1024,
# fin_time=2.0, dt_save=0.01 -> 201 saved frames including t=0)
# ---------------------------------------------------------------------------
NX = 1024
XL, XR = 0.0, 1.0
DX = (XR - XL) / NX

INI_TIME = 0.0
FIN_TIME = 2.0
DT_SAVE = 0.01
N_T = int(round((FIN_TIME - INI_TIME) / DT_SAVE)) + 1   # 201
DT_INNER = 1.0e-4
N_INNER = int(round(DT_SAVE / DT_INNER))                 # 100

X_C = jnp.linspace(XL + 0.5 * DX, XR - 0.5 * DX, NX)
T_C = jnp.arange(N_T) * DT_SAVE
K_R = 2.0 * jnp.pi * jnp.fft.rfftfreq(NX, d=DX)
K_SQ = K_R ** 2


# ---------------------------------------------------------------------------
# Vendored init_multi (PDEBench BurgersEq utils style, with if_norm=True path
# for RD). Each call with a given init_key produces ``numbers`` distinct
# initial conditions.
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def init_multi(
    xc: jnp.ndarray,
    numbers: int,
    k_tot: int = 4,
    init_key: int = 2022,
    num_choise_k: int = 2,
    if_norm: bool = True,
) -> jnp.ndarray:
    def _pass(carry):
        return carry

    def select_A(carry):
        cond, value = carry
        value = lax.cond(cond == 1, lambda c: jnp.abs(c), _pass, value)
        return cond, value

    def select_W(carry):
        def _window(c):
            xx, val, xL, xR, trns = c
            val = 0.5 * (jnp.tanh((xx - xL) / trns) - jnp.tanh((xx - xR) / trns))
            return xx, val, xL, xR, trns

        cond, value, xx, xL, xR, trns = carry
        c = (xx, value, xL, xR, trns)
        xx, value, xL, xR, trns = lax.cond(cond == 1, _window, _pass, c)
        return cond, value, xx, xL, xR, trns

    key = random.PRNGKey(init_key)
    selected = random.randint(key, [numbers, num_choise_k], 0, k_tot)
    selected = nn.one_hot(selected, k_tot, dtype=jnp.int32).sum(axis=1)
    kk = jnp.pi * 2.0 * jnp.arange(1, k_tot + 1) * selected / (xc[-1] - xc[0])
    amp = random.uniform(key, [numbers, k_tot, 1])

    key, _ = random.split(key)
    phs = 2.0 * jnp.pi * random.uniform(key, [numbers, k_tot, 1])
    _u = amp * jnp.sin(kk[:, :, None] * xc[None, None, :] + phs)
    _u = jnp.sum(_u, axis=1)

    cond = random.choice(key, 2, p=jnp.array([0.9, 0.1]), shape=[numbers])
    _, _u = vmap(select_A, 0, 0)((cond, _u))
    sgn = random.choice(key, a=jnp.array([1, -1]), shape=[numbers, 1])
    _u = _u * sgn

    key, _ = random.split(key)
    cond = random.choice(key, 2, p=jnp.array([0.9, 0.1]), shape=[numbers])
    _xc = jnp.repeat(xc[None, :], numbers, axis=0)
    mask = jnp.ones_like(_xc)
    xL = random.uniform(key, [numbers], minval=0.1, maxval=0.45)
    xR = random.uniform(key, [numbers], minval=0.55, maxval=0.9)
    trns = 0.01 * jnp.ones_like(cond, dtype=_xc.dtype)
    _, mask, _xc, xL, xR, trns = vmap(select_W, 0, 0)(
        (cond, mask, _xc, xL, xR, trns)
    )
    _u = _u * mask

    if if_norm:
        _u = _u - jnp.min(_u, axis=1, keepdims=True)
        _u = _u / (jnp.max(_u, axis=1, keepdims=True) + 1.0e-30)
    return _u


# ---------------------------------------------------------------------------
# Reaction (PDEBench piecewise-exact) and diffusion (FFT) sub-steps
# ---------------------------------------------------------------------------
def _react_exact(u: jnp.ndarray, rho: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """Exact closed-form solution of u_t = rho * u (1-u) over duration ``dt``."""
    return 1.0 / (1.0 + jnp.exp(-rho * dt) * (1.0 - u) / (u + 1.0e-30))


def _diffuse_spectral(u: jnp.ndarray, nu: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    u_hat = jnp.fft.rfft(u)
    u_hat = u_hat * jnp.exp(-nu * K_SQ * dt)
    return jnp.fft.irfft(u_hat, n=NX)


def _strang_step(u: jnp.ndarray, nu: jnp.ndarray, rho: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    u = _react_exact(u, rho, 0.5 * dt)
    u = _diffuse_spectral(u, nu, dt)
    u = _react_exact(u, rho, 0.5 * dt)
    return u


def _evolve_one(u0: jnp.ndarray, nu: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    """Solve one trajectory; return (N_T, NX) array."""

    def save_step(u, _):
        def inner(_, u_):
            return _strang_step(u_, nu, rho, DT_INNER)
        u_next = lax.fori_loop(0, N_INNER, inner, u)
        return u_next, u_next

    _, uu_rest = lax.scan(save_step, u0, jnp.arange(N_T - 1, dtype=jnp.int32))
    return jnp.concatenate([u0[None, :], uu_rest], axis=0)


@jax.jit
def solve_batch(u0s: jnp.ndarray, nus: jnp.ndarray, rhos: jnp.ndarray) -> jnp.ndarray:
    """Vectorized solver. u0s: (B, NX); nus, rhos: (B,) -> (B, N_T, NX)."""
    return vmap(_evolve_one, in_axes=(0, 0, 0))(u0s, nus, rhos)
