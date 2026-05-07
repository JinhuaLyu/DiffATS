"""1D Burgers solver — vendored copy of the PDEBench JAX FVM scheme.

Self-contained: no `pdebench` import. Exposes:
    - ``init_multi(xc, numbers, k_tot, init_key, num_choise_k, if_norm)``: random
      multi-mode sinusoid initial conditions (PDEBench BurgersEq style).
    - ``solve_batch(epsilons, u0s)``: vmapped JIT'd solver returning the full
      space-time trajectory of shape ``(B, N_T, N_X)`` in PDEBench convention.

Equation form (PDEBench):  ``u_t + u u_x = (epsilon / pi) u_xx`` on x in [0, 1]
with periodic BC. Time domain [0, FIN_TIME]; saved every ``DT_SAVE``.

Numerics: 2nd-order MUSCL-VL slope reconstruction + upwind Rusanov flux for the
convective term, central diffusion. Predictor-corrector (MacCormack-like) time
update. Adaptive dt with CFL cap on advection and diffusion separately, plus a
cap that prevents overshoot of the next save time.
"""

from __future__ import annotations

from functools import partial
from math import ceil

import jax
import jax.numpy as jnp
from jax import lax, nn, random, vmap

# ---------------------------------------------------------------------------
# Domain & solver constants (match PDEBench BurgersEq config defaults)
# ---------------------------------------------------------------------------
NX = 1024
XL, XR = 0.0, 1.0
DX = (XR - XL) / NX

INI_TIME = 0.0
FIN_TIME = 2.0
DT_SAVE = 0.01
N_T = ceil((FIN_TIME - INI_TIME) / DT_SAVE) + 1  # 201 frames including t=0

CFL = 0.25

# Per-save inner step budget. With nx=1024 and the largest viscosity
# (eps = 1e-1 -> eps/pi ~ 3.18e-2), diffusion-limited dt ~3.7e-6, so reaching
# a save interval of 0.01 needs ~2700 inner steps. We pad to 3000 for margin;
# cases with smaller viscosity finish early and the residual iterations are
# cheap no-ops because adaptive dt collapses to 0 once t reaches the target.
N_INNER = 3000

X_E = jnp.linspace(XL, XR, NX + 1)
X_C = X_E[:-1] + 0.5 * DX
T_C = jnp.arange(N_T) * DT_SAVE
PI_INV = 1.0 / jnp.pi


# ---------------------------------------------------------------------------
# Initial conditions (vendored from pdebench/data_gen/data_gen_NLE/utils.py)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def init_multi(
    xc: jnp.ndarray,
    numbers: int = 10000,
    k_tot: int = 4,
    init_key: int = 2022,
    num_choise_k: int = 2,
    if_norm: bool = False,
) -> jnp.ndarray:
    def _pass(carry):
        return carry

    def select_A(carry):
        def _abs(c):
            return jnp.abs(c)

        cond, value = carry
        value = lax.cond(cond == 1, _abs, _pass, value)
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
    selected = random.randint(
        key, shape=[numbers, num_choise_k], minval=0, maxval=k_tot
    )
    selected = nn.one_hot(selected, k_tot, dtype=jnp.int32).sum(axis=1)
    kk = jnp.pi * 2.0 * jnp.arange(1, k_tot + 1) * selected / (xc[-1] - xc[0])
    amp = random.uniform(key, shape=[numbers, k_tot, 1])

    key, _ = random.split(key)
    phs = 2.0 * jnp.pi * random.uniform(key, shape=[numbers, k_tot, 1])
    _u = amp * jnp.sin(kk[:, :, jnp.newaxis] * xc[jnp.newaxis, jnp.newaxis, :] + phs)
    _u = jnp.sum(_u, axis=1)

    cond = random.choice(key, 2, p=jnp.array([0.9, 0.1]), shape=[numbers])
    _, _u = vmap(select_A, 0, 0)((cond, _u))
    sgn = random.choice(key, a=jnp.array([1, -1]), shape=[numbers, 1])
    _u = _u * sgn

    key, _ = random.split(key)
    cond = random.choice(key, 2, p=jnp.array([0.9, 0.1]), shape=[numbers])
    _xc = jnp.repeat(xc[None, :], numbers, axis=0)
    mask = jnp.ones_like(_xc)
    xL = random.uniform(key, shape=[numbers], minval=0.1, maxval=0.45)
    xR = random.uniform(key, shape=[numbers], minval=0.55, maxval=0.9)
    trns = 0.01 * jnp.ones_like(cond, dtype=_xc.dtype)
    _, mask, _xc, xL, xR, trns = vmap(select_W, 0, 0)(
        (cond, mask, _xc, xL, xR, trns)
    )
    _u = _u * mask

    if if_norm:
        _u = _u - jnp.min(_u, axis=1, keepdims=True)
        _u = _u / jnp.max(_u, axis=1, keepdims=True)

    return _u


# ---------------------------------------------------------------------------
# Boundary, limiter, fluxes (vendored, using modern .at[] indexed-update)
# ---------------------------------------------------------------------------
def _bc_periodic(u: jnp.ndarray) -> jnp.ndarray:
    """Pad with 2 ghost cells on each side (periodic)."""
    out = jnp.zeros(NX + 4, dtype=u.dtype)
    out = out.at[2 : NX + 2].set(u)
    out = out.at[0:2].set(u[-2:])
    out = out.at[NX + 2 : NX + 4].set(u[0:2])
    return out


def _vl_limiter(a, b, c, alpha=2.0):
    return (
        jnp.sign(c)
        * (0.5 + 0.5 * jnp.sign(a * b))
        * jnp.minimum(alpha * jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    )


def _limiting(u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    du_L = u[1 : NX + 3] - u[0 : NX + 2]
    du_R = u[2 : NX + 4] - u[1 : NX + 3]
    du_M = (u[2 : NX + 4] - u[0 : NX + 2]) * 0.5
    gradu = _vl_limiter(du_L, du_R, du_M)
    uL = jnp.zeros_like(u).at[1 : NX + 3].set(u[1 : NX + 3] - 0.5 * gradu)
    uR = jnp.zeros_like(u).at[1 : NX + 3].set(u[1 : NX + 3] + 0.5 * gradu)
    return uL, uR


def _flux(u: jnp.ndarray, epsilon: jnp.ndarray) -> jnp.ndarray:
    _u = _bc_periodic(u)
    uL, uR = _limiting(_u)
    fL = 0.5 * uL ** 2
    fR = 0.5 * uR ** 2
    f = 0.5 * (
        fR[1 : NX + 2]
        + fL[2 : NX + 3]
        - 0.5
        * jnp.abs(uL[2 : NX + 3] + uR[1 : NX + 2])
        * (uL[2 : NX + 3] - uR[1 : NX + 2])
    )
    f = f - epsilon * PI_INV * (_u[2 : NX + 3] - _u[1 : NX + 2]) / DX
    return f


def _update(u: jnp.ndarray, u_tmp: jnp.ndarray, epsilon: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    f = _flux(u_tmp, epsilon)
    return u - dt / DX * (f[1 : NX + 1] - f[0:NX])


def _macstep(u: jnp.ndarray, epsilon: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    u_tmp = _update(u, u, epsilon, 0.5 * dt)
    return _update(u, u_tmp, epsilon, dt)


def _courant(u: jnp.ndarray) -> jnp.ndarray:
    return DX / (jnp.max(jnp.abs(u)) + 1.0e-8)


def _courant_diff(epsilon: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * DX ** 2 / (epsilon * PI_INV + 1.0e-8)


# ---------------------------------------------------------------------------
# Per-trajectory time integration (no Python while_loop, vmap-friendly)
# ---------------------------------------------------------------------------
def _evolve_one(epsilon: jnp.ndarray, u0: jnp.ndarray) -> jnp.ndarray:
    """Solve one trajectory; return array of shape (N_T, NX) in PDEBench order."""

    def inner_step(_, st):
        u, t, target_t = st
        dt_adv = _courant(u) * CFL
        dt_dif = _courant_diff(epsilon) * CFL
        dt = jnp.minimum(jnp.minimum(dt_adv, dt_dif), target_t - t)
        dt = jnp.where(dt < 1.0e-12, 0.0, dt)
        u_new = jnp.where(dt > 1.0e-12, _macstep(u, epsilon, dt), u)
        return (u_new, t + dt, target_t)

    def save_step(carry, i):
        u, t = carry
        target_t = (i + 1).astype(u0.dtype) * DT_SAVE
        u, t, _ = lax.fori_loop(0, N_INNER, inner_step, (u, t, target_t))
        return (u, t), u

    init_carry = (u0, jnp.asarray(INI_TIME, dtype=u0.dtype))
    (_u_final, _t_final), uu_rest = lax.scan(
        save_step, init_carry, jnp.arange(N_T - 1, dtype=jnp.int32)
    )
    uu = jnp.concatenate([u0[None, :], uu_rest], axis=0)
    return uu


@jax.jit
def solve_batch(epsilons: jnp.ndarray, u0s: jnp.ndarray) -> jnp.ndarray:
    """Vectorized solver. epsilons: (B,), u0s: (B, NX) -> (B, N_T, NX)."""
    return vmap(_evolve_one, in_axes=(0, 0))(epsilons, u0s)
