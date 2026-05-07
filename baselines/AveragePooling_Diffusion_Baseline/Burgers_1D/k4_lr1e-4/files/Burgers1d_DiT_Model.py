import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D Sinusoidal Positional Embedding
def get_1d_sincos_pos_embed(embed_dim, length):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim // 2)
    omega = 1.0 / (10000 ** omega)
    pos   = torch.arange(length, dtype=torch.float32)
    out   = torch.outer(pos, omega)   # [length, embed_dim//2]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [length, embed_dim]


# Timestep Embedder (diffusion noise step t)
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        half  = self.freq_dim // 2
        freqs = torch.arange(half, device=t.device).float() / half
        freqs = 1.0 / (10000 ** freqs)
        x = t[:, None].float() * freqs[None]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(x)   # [B, hidden_dim]



# Nu Scalar Embedder

class NuEmbedder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, nu):
        """nu: [B, 1]"""
        return self.mlp(nu)   # [B, hidden_dim]



# DiT Block (AdaLN-Zero)
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        mlp_dim    = int(hidden_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        xn = self.norm1(x) * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        a, _ = self.attn(xn, xn, xn)
        x = x + g1.unsqueeze(1) * a
        xn = self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(xn)
        return x


# Final Layer

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


# 1D TrajDiT

class TrajDiT1D(nn.Module):
    def __init__(
        self,
        L_lr=256,          # spatial length after pooling
        num_frames=200,    # target timesteps
        hidden_dim=512,
        num_heads=8,
        num_layers=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.L_lr       = L_lr
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # target token embedder: 200 frames -> hidden_dim
        self.target_embedder = nn.Linear(num_frames, hidden_dim)

        # condition spatial embedder: 1 frame -> hidden_dim (1 token)
        self.cond_embedder = nn.Linear(1, hidden_dim)

        # 1D positional embedding for L_lr + 1 (cond token)
        pos_embed = get_1d_sincos_pos_embed(hidden_dim, L_lr + 1)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))  # [1, L_lr+1, D]

        # diffusion timestep embedder
        self.t_embedder  = TimestepEmbedder(hidden_dim)

        # nu scalar embedder
        self.nu_embedder = NuEmbedder(hidden_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        # final layer: each target token predicts 200 values
        self.final_layer = FinalLayer(hidden_dim, num_frames)

    def forward(self, noisy_target, cond_spatial, cond_nu, t):
        B, T, L = noisy_target.shape

        # tokenise target: [B, 200, L_lr] -> [B, L_lr, 200] -> embed -> [B, L_lr, D]
        target_tokens = noisy_target.permute(0, 2, 1)          # [B, L_lr, 200]
        target_tokens = self.target_embedder(target_tokens)     # [B, L_lr, D]

        # tokenise condition spatial: [B, 1, L_lr] -> [B, L_lr, 1] -> embed -> [B, L_lr, D]
        # use only the single cond frame, then prepend as 1 extra token
        cond_tok = cond_spatial.permute(0, 2, 1)               # [B, L_lr, 1]
        cond_tok = self.cond_embedder(cond_tok)                 # [B, L_lr, D]
        # take mean over spatial to get 1 token representing t=0
        cond_token = cond_tok.mean(dim=1, keepdim=True)        # [B, 1, D]

        # concatenate: [cond_token | target_tokens] -> [B, L_lr+1, D]
        tokens = torch.cat([cond_token, target_tokens], dim=1) # [B, L_lr+1, D]

        # add positional embedding
        tokens = tokens + self.pos_embed                        # [B, L_lr+1, D]

        # conditioning: diffusion t + nu
        c = self.t_embedder(t) + self.nu_embedder(cond_nu)     # [B, D]

        # transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # final layer on target tokens only (skip cond token)
        target_out = self.final_layer(tokens[:, 1:, :], c)     # [B, L_lr, 200]

        # reshape back: [B, L_lr, 200] -> [B, 200, L_lr]
        return target_out.permute(0, 2, 1)


# Gaussian Diffusion (DDPM + DDIM)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model     = model
        self.timesteps = timesteps

        betas               = torch.linspace(beta_start, beta_end, timesteps)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas',                         betas)
        self.register_buffer('alphas_cumprod',                alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',           alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',           alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1-alphas_cumprod).sqrt())

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alphas_cumprod[t][:, None, None]
        sb = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sa * x0 + sb * noise, noise

    def training_loss(self, target, cond_spatial, cond_nu):
        B  = target.shape[0]
        t  = torch.randint(0, self.timesteps, (B,), device=target.device)
        noisy_x, noise = self.q_sample(target, t)
        pred_noise = self.model(noisy_x, cond_spatial, cond_nu, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, cond_spatial, cond_nu, num_steps=250, eta=0.0):
        device = next(self.model.parameters()).device
        B      = cond_spatial.shape[0]
        T      = self.model.num_frames
        L      = self.model.L_lr

        x = torch.randn(B, T, L, device=device)

        step_size = self.timesteps // num_steps
        timesteps = list(reversed(range(0, self.timesteps, step_size)))

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            pred_noise = self.model(x, cond_spatial, cond_nu, t)

            alpha      = self.alphas_cumprod[t_val]
            alpha_prev = self.alphas_cumprod[timesteps[i+1]] \
                         if i+1 < len(timesteps) else torch.tensor(1.0)

            pred_x0 = (x - (1-alpha).sqrt() * pred_noise) / alpha.sqrt()
            pred_x0 = pred_x0.clamp(-1, 1)

            sigma = eta * ((1-alpha_prev)/(1-alpha) * (1-alpha/alpha_prev)).sqrt()
            noise = torch.randn_like(x) if eta > 0 else 0
            x = (alpha_prev.sqrt() * pred_x0
                 + (1-alpha_prev-sigma**2).clamp(min=0).sqrt() * pred_noise
                 + sigma * noise)

        return x   # [B, 200, L_lr]


if __name__ == '__main__':
    model = TrajDiT1D(
        L_lr=256, num_frames=200,
        hidden_dim=512, num_heads=8, num_layers=12,
    )
    diffusion = GaussianDiffusion(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params/1e6:.2f}M")
    print(f"Tokens     : {model.L_lr + 1}  ({model.L_lr} target + 1 cond)")

    B = 4
    target       = torch.randn(B, 200, 256)
    cond_spatial = torch.randn(B, 1, 256)
    cond_nu      = torch.rand(B, 1)

    loss = diffusion.training_loss(target, cond_spatial, cond_nu)
    print(f"Train loss : {loss.item():.4f}")

    gen = diffusion.ddim_sample(cond_spatial, cond_nu, num_steps=10)
    print(f"Generated  : {gen.shape}")   # [B, 200, 256]
    print("Model ready.")
