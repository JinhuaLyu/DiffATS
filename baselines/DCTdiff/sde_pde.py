

import torch
import numpy as np
from tqdm import tqdm

from sde import VPSDE, ScoreModel, ReverseSDE, ODE, stp, duplicate


def _broadcast_mask(mask, like):
    # mask: (n_tokens,) bool ; like: (B, n_tokens, F)
    return mask.view(1, -1, 1).expand_as(like)


class CondScoreModel(ScoreModel):

    def __init__(self, nnet, pred, sde, T=1, n_ic_tokens=0):
        super().__init__(nnet, pred, sde, T)
        self.n_ic_tokens = n_ic_tokens

    def predict(self, xt, t, nu=None, ic_clean=None, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))

        if ic_clean is not None and self.n_ic_tokens > 0:
            xt = xt.clone()
            xt[:, : self.n_ic_tokens] = ic_clean

        return self.nnet(xt, t * 999, nu=nu, **kwargs)


def LSimple_masked(score_model: CondScoreModel, x0, nu, ic_clean,
                   ic_mask, pred='noise_pred', reweight=None):
    """Diffusion training loss restricted to non-IC tokens."""
    t, noise, xt = score_model.sde.sample(x0)

    # Force the IC slice of xt to be the *clean* IC at every timestep so
    # the network always sees a noise-free initial condition.
    if score_model.n_ic_tokens > 0:
        xt = xt.clone()
        xt[:, : score_model.n_ic_tokens] = ic_clean

    if pred == 'noise_pred':
        pred_t = score_model.noise_pred(xt, t, nu=nu, ic_clean=ic_clean)
        diff = (noise - pred_t).pow(2)
    elif pred == 'x0_pred':
        x0_pred = score_model.x0_pred(xt, t, nu=nu, ic_clean=ic_clean)
        diff = (x0 - x0_pred).pow(2)
    else:
        raise NotImplementedError(pred)

    if reweight is not None:
        diff = diff * reweight                                         # (B, L, F)

    keep = (~ic_mask).to(diff.device).view(1, -1, 1).float()           # (1, L, 1)
    n_keep = keep.sum() * diff.shape[2]                                # scalar = L_keep * F
    return (diff * keep).flatten(1).sum(dim=-1) / n_keep                # (B,)


@torch.no_grad()
def euler_maruyama_cond(rsde, x_init, sample_steps, n_ic_tokens, ic_clean,
                        eps=1e-3, T=1, **kwargs):
    """Reverse SDE / ODE sampler that pins the IC slice at every step."""
    print(f"euler_maruyama_cond steps={sample_steps}")
    timesteps = np.append(0., np.linspace(eps, T, sample_steps))
    timesteps = torch.tensor(timesteps).to(x_init)
    x = x_init.clone()
    if n_ic_tokens > 0:
        x[:, :n_ic_tokens] = ic_clean

    for s, t in list(zip(timesteps, timesteps[1:]))[::-1]:
        drift = rsde.drift(x, t, ic_clean=ic_clean, **kwargs)
        diffusion = rsde.diffusion(t)
        dt = s - t
        mean = x + drift * dt
        sigma = diffusion * (-dt).sqrt()
        x = mean + stp(sigma, torch.randn_like(x)) if s != 0 else mean
        if n_ic_tokens > 0:
            x[:, :n_ic_tokens] = ic_clean
    return x
