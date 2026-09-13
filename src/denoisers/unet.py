"""
1D conditional U-Net for time-series diffusion (DDPM), wrapped in a LightningModule.

Shapes convention:
    x : (B, N)            - batch of target vectors (time series of length N)
    c : (B, context_dim)  - batch of context vectors
    k : (B,)  long         - batch of diffusion timesteps in [0, T-1]

The U-Net itself is a plain nn.Module (architecture) kept separate from the
LightningModule (training loop / diffusion math), so each can be unit-tested
independently.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Standard transformer-style sinusoidal embedding for a scalar (the timestep k)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        # k: (B,) integer or float tensor
        device = k.device
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq)
        args = k.float()[:, None] * freq[None, :]          # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb


class MLPEmbedding(nn.Module):
    """Small MLP used both for the timestep embedding and the context embedding."""

    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Residual block with FiLM conditioning (scale/shift from the embedding)
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        # Projects the conditioning embedding to a per-channel scale and shift (FiLM).
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)   # each (B, out_ch)
        scale = scale[:, :, None]
        shift = shift[:, :, None]
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))

        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.block = ResBlock1D(in_ch, out_ch, emb_dim)
        self.downsample = nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, emb):
        h = self.block(x, emb)
        return self.downsample(h), h  # downsampled output, skip connection


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        self.block = ResBlock1D(in_ch + skip_ch, out_ch, emb_dim)

    def forward(self, x, skip, emb):
        h = self.upsample(x)
        if h.shape[-1] != skip.shape[-1]:  # guard against odd-length rounding
            h = F.interpolate(h, size=skip.shape[-1])
        h = torch.cat([h, skip], dim=1)
        return self.block(h, emb)


# ---------------------------------------------------------------------------
# The U-Net
# ---------------------------------------------------------------------------

class UNet1D(nn.Module):
    def __init__(
        self,
        context_dim: int,
        base_ch: int = 64,
        ch_mults: tuple = (1, 2, 4),
        emb_dim: int = 256,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(SinusoidalPosEmb(base_ch), MLPEmbedding(base_ch, emb_dim))
        self.context_embed = MLPEmbedding(context_dim, emb_dim)

        self.init_conv = nn.Conv1d(1, base_ch, kernel_size=3, padding=1)

        chs = [base_ch * m for m in ch_mults]
        self.downs = nn.ModuleList()
        in_ch = base_ch
        for out_ch in chs:
            self.downs.append(Down(in_ch, out_ch, emb_dim))
            in_ch = out_ch

        self.bottleneck1 = ResBlock1D(in_ch, in_ch, emb_dim)
        self.bottleneck2 = ResBlock1D(in_ch, in_ch, emb_dim)

        self.ups = nn.ModuleList()
        for out_ch in reversed(chs[:-1] + (base_ch,)):
            self.ups.append(Up(in_ch, out_ch, out_ch, emb_dim))
            in_ch = out_ch

        self.final_norm = nn.GroupNorm(min(8, base_ch), base_ch)
        self.final_conv = nn.Conv1d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, k: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, N) -> (B, 1, N)
        x = x.unsqueeze(1)

        emb = self.time_embed(k) + self.context_embed(c)   # (B, emb_dim)

        h = self.init_conv(x)
        skips = []
        for down in self.downs:
            h, skip = down(h, emb)
            skips.append(skip)

        h = self.bottleneck1(h, emb)
        h = self.bottleneck2(h, emb)

        for up in self.ups:
            h = up(h, skips.pop(), emb)

        out = self.final_conv(F.silu(self.final_norm(h)))
        return out.squeeze(1)  # (B, N), predicted noise


# ---------------------------------------------------------------------------
# Diffusion process (kept separate from the LightningModule so it's testable
# without a Trainer)
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device="cpu"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, k: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: sample x_k given x0 (closed form)."""
        sqrt_ac = self.sqrt_alphas_cumprod[k][:, None]
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[k][:, None]
        return sqrt_ac * x0 + sqrt_1m_ac * noise


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class DDPMLightningModule(L.LightningModule):
    def __init__(
        self,
        context_dim: int,
        base_ch: int = 64,
        timesteps: int = 1000,
        lr: float = 2e-4,
        cfg_dropout: float = 0.1,  # classifier-free guidance: randomly drop context during training
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = UNet1D(context_dim=context_dim, base_ch=base_ch)
        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        self.cfg_dropout = cfg_dropout

    def training_step(self, batch, batch_idx):
        x0, c = batch["x"], batch["c"]           # x0: (B, N), c: (B, context_dim)
        B = x0.shape[0]

        # Classifier-free guidance: zero out context on a random subset of the batch
        # so the model also learns the unconditional case.
        if self.cfg_dropout > 0:
            mask = (torch.rand(B, device=self.device) < self.cfg_dropout).float()[:, None]
            c = c * (1 - mask)

        k = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x0)

        # move diffusion buffers to the right device lazily
        self.diffusion.sqrt_alphas_cumprod = self.diffusion.sqrt_alphas_cumprod.to(self.device)
        self.diffusion.sqrt_one_minus_alphas_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod.to(self.device)

        x_k = self.diffusion.q_sample(x0, k, noise)
        pred_noise = self.unet(x_k, k, c)

        loss = F.mse_loss(pred_noise, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, c = batch["x"], batch["c"]
        B = x0.shape[0]
        k = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_k = self.diffusion.q_sample(x0, k, noise)
        pred_noise = self.unet(x_k, k, c)
        loss = F.mse_loss(pred_noise, noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

