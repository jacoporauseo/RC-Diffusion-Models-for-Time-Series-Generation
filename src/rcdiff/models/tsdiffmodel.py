import lightning as pl
import torch 
import torch.nn as nn 
from rcdiff.diff.diffusion import NoiseProcess 
import torch.nn.functional as F

class DiffusionLitModule(pl.LightningModule):
    def __init__(self, 
                 encoder: nn.Module, 
                 denoiser: nn.Module, 
                 diffuser : NoiseProcess
                 ):
        super().__init__()
        self.encoder = encoder      # registered submodule
        self.denoiser = denoiser    # registered submodule
        self.diffuser = diffuser            # NOT an nn.Module -> see below
        self.save_hyperparameters(ignore=["encoder", "denoiser", "ddpm"])

    def training_step(self, batch, batch_idx):
        x_0, history = batch
        s = self.encoder(history)
        k = torch.randint(0, self.diffuser.scheduler.K, size = (x_0.shape[0],), device=x_0.device)
        x_k, eps = self.diffuser.q_sample(x_0, k)
        eps_hat = self.denoiser(x_k, k, y=s)
        loss = F.mse_loss(eps_hat, eps)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.denoiser.parameters()),
            lr=1e-3,
        )