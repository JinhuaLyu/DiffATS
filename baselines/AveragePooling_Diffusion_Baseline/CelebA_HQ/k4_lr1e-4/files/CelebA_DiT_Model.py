import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional Embedding

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid   = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid   = torch.stack(grid, dim=0).reshape(2, -1)

    assert embed_dim % 4 == 0
    half  = embed_dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)

    def encode(pos):
        out = torch.outer(pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_h = encode(grid[0])
    emb_w = encode(grid[1])
    return torch.cat([emb_h, emb_w], dim=1)   # [grid_size^2, D]


# Timestep Embedder

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
        return self.mlp(x)


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
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
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

# Unconditional DiT for CelebA-HQ

class CelebADiT(nn.Module):

    def __init__(
        self,
        lr_size=256,        # spatial size after avg pool
        patch_size=8,
        in_channels=3,      
        hidden_dim=1024,
        num_heads=16,
        num_layers=24,
        mlp_ratio=4.0,
    ):
        super().__init__()
        assert lr_size % patch_size == 0
        self.lr_size     = lr_size
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.hidden_dim  = hidden_dim
        self.num_patches = (lr_size // patch_size) ** 2   # 1024 for 256/8

        # patch embedder
        self.patch_embedder = nn.Linear(
            patch_size * patch_size * in_channels, hidden_dim
        )

        # positional embedding
        pos_embed = get_2d_sincos_pos_embed(hidden_dim, lr_size // patch_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))  # [1, N, D]

        # diffusion timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        # final layer
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels)

    def patchify(self, x):
        
        p = self.patch_size
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)      # [B, H/p, W/p, C, p, p]
        x = x.reshape(B, -1, C * p * p)       # [B, N, C*p*p]
        return x

    def unpatchify(self, x):
        
        p  = self.patch_size
        C  = self.in_channels
        hw = int(x.shape[1] ** 0.5)
        x  = x.reshape(x.shape[0], hw, hw, C, p, p)
        x  = x.permute(0, 3, 1, 4, 2, 5)     # [B, C, hw, p, hw, p]
        x  = x.reshape(x.shape[0], C, hw*p, hw*p)
        return x

    def forward(self, noisy_image, t):
        
        # patchify
        tokens = self.patchify(noisy_image)            # [B, N, p*p*C]
        tokens = self.patch_embedder(tokens) + self.pos_embed  # [B, N, D]

        # conditioning: diffusion timestep only
        c = self.t_embedder(t)   # [B, D]

        # transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # final layer
        tokens = self.final_layer(tokens, c)           # [B, N, p*p*C]
        return self.unpatchify(tokens)                 # [B, 3, lr_size, lr_size]


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

        self.register_buffer('betas',                        betas)
        self.register_buffer('alphas_cumprod',               alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',          alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',          alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',(1-alphas_cumprod).sqrt())

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sb = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sa * x0 + sb * noise, noise

    def training_loss(self, x0):
    
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noisy_x, noise = self.q_sample(x0, t)
        pred_noise = self.model(noisy_x, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, batch_size, num_steps=250, eta=0.0):

        device  = next(self.model.parameters()).device
        C       = self.model.in_channels
        H = W   = self.model.lr_size

        x = torch.randn(batch_size, C, H, W, device=device)

        step_size = self.timesteps // num_steps
        timesteps = list(reversed(range(0, self.timesteps, step_size)))

        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            pred_noise = self.model(x, t)

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

        return x   # [B, 3, lr_size, lr_size]


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--lr_size',    type=int, default=256)
    p.add_argument('--patch_size', type=int, default=8)
    p.add_argument('--hidden_dim', type=int, default=1024)
    p.add_argument('--num_layers', type=int, default=24)
    p.add_argument('--num_heads',  type=int, default=16)
    args = p.parse_args()

    model = CelebADiT(
        lr_size=args.lr_size,
        patch_size=args.patch_size,
        in_channels=3,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    diffusion = GaussianDiffusion(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params/1e6:.2f}M")
    print(f"Tokens     : {model.num_patches}  ({args.lr_size//args.patch_size}x{args.lr_size//args.patch_size})")

    B = 2
    img  = torch.randn(B, 3, args.lr_size, args.lr_size)
    loss = diffusion.training_loss(img)
    print(f"Train loss : {loss.item():.4f}")

    gen = diffusion.ddim_sample(batch_size=2, num_steps=10)
    print(f"Generated  : {gen.shape}")
    print("Model ready.")
