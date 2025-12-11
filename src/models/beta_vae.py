import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.brain_tumor_utils.config_parser import get_config
from training.losses import FocalFrequencyLoss, LPIPSLoss
from .se_blocks import SEBlock

def _get_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError("unsupported activation")

def _get_norm(name, ch):
    if name == "batch":
        return nn.BatchNorm2d(ch)
    if name == "layer":
        return nn.GroupNorm(1, ch)
    if name == "none":
        return nn.Identity()
    raise ValueError("unsupported norm")

class SEBlockWrapper(nn.Module):
    def __init__(self, channels, use_se, reduction):
        super().__init__()
        self.block = SEBlock(channels, reduction) if use_se else nn.Identity()
    def forward(self, x):
        return self.block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type, activation, use_se, se_reduction, down=True):
        super().__init__()
        stride = 2 if down else 1
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = _get_norm(norm_type, out_ch)
        self.act = activation
        self.se = SEBlockWrapper(out_ch, use_se, se_reduction)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.se(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type, activation, use_se, se_reduction, up=True):
        super().__init__()
        if up:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.up = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = _get_norm(norm_type, out_ch)
        self.act = activation
        self.se = SEBlockWrapper(out_ch, use_se, se_reduction)
    def forward(self, x):
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.se(x)
        return x

class BetaVAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or get_config()
        mcfg = self.cfg.model
        dcfg = self.cfg.data
        self.latent_dim = mcfg.latent_dim
        base = mcfg.base_channels
        blocks = mcfg.num_blocks
        act = _get_activation(mcfg.activation)
        norm_type = mcfg.encoder_norm
        se_reduction = mcfg.se_reduction_ratio
        use_dec_se = mcfg.use_decoder_se
        in_ch = 1 if dcfg.grayscale else 3
        self.deterministic = getattr(mcfg, "deterministic_overfit", False)
        self.latent_reg_lambda = getattr(mcfg, "latent_reg_lambda", 0.0)
        self.pooling = getattr(mcfg, "encoder_pooling", "flatten")
        self.latent_clamp = getattr(mcfg, "latent_clamp", None)
        lcfg = getattr(self.cfg, "loss", None)
        self.use_lpips = getattr(lcfg, "use_lpips", False)
        self.use_ffl = getattr(lcfg, "use_ffl", False)
        self.lpips_weight = getattr(lcfg, "lpips_weight", 0.0)
        self.ffl_weight = getattr(lcfg, "ffl_weight", 0.0)
        self.lpips_net_name = getattr(lcfg, "lpips_net", "alex")
        self.lpips_loss = None
        self.ffl_loss = FocalFrequencyLoss(alpha=getattr(lcfg, "ffl_alpha", 1.0))
        chs = [in_ch]
        for i in range(blocks):
            chs.append(base * (2 ** i))
        enc_layers = []
        for i in range(blocks):
            enc_layers.append(ConvBlock(chs[i], chs[i+1], norm_type, act, True, se_reduction, down=True))
        self.encoder = nn.Sequential(*enc_layers)
        pool_type = getattr(mcfg, "encoder_pooling", "flatten")
        if pool_type == "gap":
            self.enc_pool = nn.AdaptiveAvgPool2d((1,1))
        elif pool_type == "flatten":
            self.enc_pool = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder_pooling {pool_type}")
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, dcfg.image_size, dcfg.image_size)
            h = self.encoder(dummy)
            self.enc_out_shape = h.shape[1:]
            if self.pooling == "gap":
                flat_dim = self.enc_out_shape[0]
            else:
                flat_dim = h.view(1, -1).shape[1]
        self.flat_dim = flat_dim
        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, flat_dim)
        dec_chs = list(reversed(chs[1:]))
        dec_layers = []
        for i in range(blocks):
            in_c = dec_chs[i]
            out_c = dec_chs[i+1] if i+1 < len(dec_chs) else dec_chs[-1]
            dec_layers.append(DeconvBlock(in_c, out_c, norm_type, act, use_dec_se, se_reduction, up=True))
        self.decoder_blocks = nn.Sequential(*dec_layers)
        self.final_conv = nn.Conv2d(dec_chs[-1], in_ch, 3, padding=1)
        self.current_beta = mcfg.beta
        self.recon_loss_type = mcfg.reconstruction_loss
        self.logvar_clamp = getattr(mcfg, "logvar_clamp", None)

    def set_beta(self, beta):
        self.current_beta = beta

    def encode(self, x):
        h = self.encoder(x)
        h = self.enc_pool(h)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.logvar_clamp:
            logvar = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])
        else:
            logvar = logvar.clamp_(-10, 10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        if self.latent_clamp is not None:
            z = torch.clamp(z, -self.latent_clamp, self.latent_clamp)
        h = self.fc_dec(z)
        if self.pooling == "gap":
            c = self.enc_out_shape[0]
            s = self.enc_out_shape[1]
            h = h.view(z.size(0), c, 1, 1).expand(-1, c, s, s)
        else:
            h = h.view(z.size(0), *self.enc_out_shape)
        h = self.decoder_blocks(h)
        x = self.final_conv(h)
        x = torch.sigmoid(x)
        return x

    def forward(self, x, deterministic: bool = None):
        """
        Forward pass returning reconstruction and latent stats.
        deterministic:
           - If True: use mu directly (no sampling).
           - If False: always sample (even if beta==0) so KL statistics remain meaningful.
           - If None: defaults to self.deterministic flag.
        """
        if deterministic is None:
            deterministic = self.deterministic
        mu, logvar = self.encode(x)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def reconstruction_loss(self, recon, x):
        if self.recon_loss_type == "mse":
            return F.mse_loss(recon, x, reduction="sum") / x.size(0)
        if self.recon_loss_type == "bce":
            return F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
        if self.recon_loss_type == "l1":
            return F.l1_loss(recon, x, reduction="sum") / x.size(0)
        raise ValueError("invalid reconstruction_loss")

    def loss(self, x, beta=None, capacity=None, free_bits=0.0, capacity_weight=None):
        """
        Extended loss supporting:
          - beta scheduling
          - capacity (|KL - C|)
          - free bits (per-dim clamp)
        """
        recon, mu, logvar, z = self.forward(x, deterministic=self.deterministic)
        base_recon = self.reconstruction_loss(recon, x)

        lp = torch.zeros((), device=x.device)
        ff = torch.zeros((), device=x.device)

        if self.use_lpips and self.lpips_weight > 0:
            if self.lpips_loss is None:
                self.lpips_loss = LPIPSLoss(net=self.lpips_net_name).to(x.device)
            else:
                self.lpips_loss = self.lpips_loss.to(x.device)
            lp = self.lpips_loss(recon, x) * self.lpips_weight

        if self.use_ffl and self.ffl_weight > 0:
            ff = self.ffl_loss(recon, x) * self.ffl_weight

        rec_loss = base_recon + lp + ff

        if self.deterministic:
            kl_elem = torch.zeros((x.size(0), self.latent_dim), device=x.device)
            kl_per_dim = kl_elem.mean(dim=0)
            kl_mean = torch.zeros((), device=x.device)
        else:
            kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = kl_elem.mean(dim=0)
            kl_mean = kl_elem.sum(dim=1).mean()
        use_capacity = (capacity is not None) and (capacity_weight is not None)
        if not self.deterministic:
            if (free_bits > 0) and not use_capacity:
                kl_per_dim_eff = torch.clamp(kl_per_dim, min=free_bits)
                kl_effective = kl_per_dim_eff.sum()
            else:
                kl_effective = kl_per_dim.sum()
        else:
            kl_effective = torch.zeros((), device=x.device)

        b = self.current_beta if beta is None else beta
        latent_reg = 0.0
        if self.latent_reg_lambda > 0:
            latent_reg = self.latent_reg_lambda * torch.mean(mu.pow(2))

        if not self.deterministic:
            if use_capacity:
                if capacity_weight is None:
                    cfg_loss = getattr(self.cfg, "loss", None)
                    capacity_weight = getattr(cfg_loss, "capacity_weight", None) if cfg_loss else None
                gamma = capacity_weight if capacity_weight is not None else (b if b is not None else 1.0)
                kl_term = gamma * torch.abs(kl_mean - capacity)
                total = rec_loss + kl_term + latent_reg
            else:
                total = rec_loss + b * kl_effective + latent_reg
        else:
            total = rec_loss + latent_reg

        return {
            "total": total,
            "recon": rec_loss,
            "recon_base": base_recon.detach(),
            "recon_lpips": lp.detach(),
            "recon_ffl": ff.detach(),
            "kl_mean": kl_mean,
            "kl_per_dim": kl_per_dim.detach(),
            "beta": torch.tensor(b),
            "capacity": torch.tensor(capacity if capacity is not None else float('nan')),
            "latent_reg": torch.tensor(latent_reg),
            "recon_img": recon.detach(),
            "z": z.detach(),
            "mu": mu.detach(),
            "logvar": logvar.detach(),
            "kl_effective": kl_effective.detach(),
            "mode": "capacity" if use_capacity else "beta"
        }

    def sample_prior(self, n):
        z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        return self.decode(z)

    def traverse(self, x, dim, steps=7, span=3.0):
        mu, logvar = self.encode(x)
        base = mu
        outputs = []
        vals = torch.linspace(-span, span, steps, device=base.device)
        for v in vals:
            z = base.clone()
            z[:, dim] = v
            outputs.append(self.decode(z))
        return torch.stack(outputs, dim=1), vals
