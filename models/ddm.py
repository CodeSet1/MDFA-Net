import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils

from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pytorch_msssim import ssim

from models.unet import DiffusionUNet
from models.mdfa import MDFA
from models.lam import LAM
from models.acfg import ACFG


# =========================
# utils
# =========================
def data_transform(x):
    return 2 * x - 1.0


def inverse_data_transform(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


# =========================
# TV Loss (unchanged)
# =========================
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        b, c, h, w = x.size()
        count_h = c * (h - 1) * w
        count_w = c * h * (w - 1)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b


# =========================
# EMA (unchanged)
# =========================
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for n, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for n, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[n].data = (1 - self.mu) * p.data + self.mu * self.shadow[n].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for n, p in module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# =========================
# beta schedule (unchanged)
# =========================
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        return beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)


# =====================================================
# DFA-Net (core)
# =====================================================
class Net(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        # ---- DTCWT ----
        self.dtcwt = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.idtcwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        # ---- DFA-Net modules ----
        self.mdfa = MDFA(in_channels=3, hidden_dim=64, num_directions=6)
        self.lam = LAM(in_channels=3)
        self.acfg = ACFG(channels=3, gamma_init=config.model.gamma_init)

        # ---- Diffusion backbone ----
        self.Unet = DiffusionUNet(config)

        # ---- diffusion params ----
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
        return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)

    # -------------------------------------------------
    # DDIM-style sampling (unchanged logic)
    # -------------------------------------------------
    def sample_training(self, cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        n, c, h, w = cond.shape
        x = torch.randn(n, c, h, w, device=cond.device)

        for i in reversed(seq):
            t = torch.full((n,), i, device=cond.device, dtype=torch.long)
            at = self.compute_alpha(b, t)

            et = self.Unet(torch.cat([cond, x], dim=1), t.float())
            x0 = (x - et * (1 - at).sqrt()) / at.sqrt()
            x = at.sqrt() * x0 + (1 - at).sqrt() * et

        return x

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def forward(self, x):
        data = {}
        input_img = x[:, :3]
        gt_img = x[:, 3:] if self.training else None

        # ---- DTCWT decomposition ----
        x_norm = data_transform(input_img)
        yl, yh = self.dtcwt(x_norm)

        # ---- DFA-Net enhancement ----
        yl = self.lam(yl)
        yh = [self.mdfa(band) for band in yh]

        b = self.betas.to(x.device)

        t = torch.randint(
            0, self.num_timesteps,
            (yl.shape[0],),
            device=x.device
        )
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(yl)

        if self.training:
            # ---- GT low-frequency ----
            gt_norm = data_transform(gt_img)
            gt_yl, _ = self.dtcwt(gt_norm)

            x_noisy = gt_yl * a.sqrt() + e * (1 - a).sqrt()

            # ---- ACFG conditioned diffusion ----
            cond = self.acfg(yl, yh)
            noise_pred = self.Unet(torch.cat([cond, x_noisy], dim=1), t.float())

            # ---- sampling ----
            denoise_yl = self.sample_training(cond, b)

            # ---- reconstruction ----
            pred_x = self.idtcwt((denoise_yl, yh))
            pred_x = inverse_data_transform(pred_x)

            data["noise_output"] = noise_pred
            data["e"] = e
            data["pred_x"] = pred_x
            data["gt"] = gt_img
        else:
            cond = self.acfg(yl, yh)
            denoise_yl = self.sample_training(cond, b)
            pred_x = self.idtcwt((denoise_yl, yh))
            data["pred_x"] = inverse_data_transform(pred_x)

        return data


# =====================================================
# Trainer (almost unchanged)
# =====================================================
class DenoisingDiffusion(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config).to(self.device)
        self.model = nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(
            self.config, self.model.parameters()
        )
        self.step = 0

    # -------------------------------------------------
    # loss (论文语义一致)
    # -------------------------------------------------
    def estimation_loss(self, x, out):
        gt = out["gt"]
        pred = out["pred_x"]

        noise_loss = self.l2_loss(out["noise_output"], out["e"])
        photo_loss = self.l1_loss(pred, gt) + (1 - ssim(pred, gt, data_range=1.0))
        frequency_loss = 0.0  # directional info already in MDFA

        return noise_loss, photo_loss, frequency_loss

    # -------------------------------------------------
    # training loop (unchanged)
    # -------------------------------------------------
    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        for epoch in range(self.config.training.n_epochs):
            print(f"Epoch {epoch}")
            for x, _ in train_loader:
                x = x.to(self.device)
                self.model.train()
                self.step += 1

                out = self.model(x)
                loss = sum(self.estimation_loss(x, out))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)

            self.scheduler.step()
