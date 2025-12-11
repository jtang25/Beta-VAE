import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """
    Frequency-domain loss that up-weights spectra with large discrepancies.
    Uses float32 and disables autocast for stability under AMP.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        pred_f = pred.float()
        target_f = target.float()

        with torch.amp.autocast("cuda", enabled=False):
            pred_fft = torch.fft.fft2(pred_f, norm="ortho")
            target_fft = torch.fft.fft2(target_f, norm="ortho")

            diff = pred_fft - target_fft
            dist = diff.real.pow(2) + diff.imag.pow(2)

            denom = dist.mean(dim=(0, 2, 3), keepdim=True) + self.eps
            weight = (dist / denom).clamp_min(self.eps).pow(self.alpha)

            loss = (weight * dist).mean()

        return loss


class LPIPSLoss(nn.Module):
    """
    Thin wrapper around the official LPIPS implementation.
    Converts grayscale -> 3ch and scales inputs to [-1, 1] before evaluation.
    """

    def __init__(self, net: str = "alex", normalize: bool = True):
        super().__init__()
        try:
            import lpips  # type: ignore
        except ImportError as exc:
            raise ImportError("lpips is required for LPIPSLoss. Please install the 'lpips' package.") from exc
        self.normalize = normalize
        self.loss_fn = lpips.LPIPS(net=net)
        self.loss_fn.eval()
        for p in self.loss_fn.parameters():
            p.requires_grad = False

    def _ensure_device(self, x: torch.Tensor):
        if next(self.loss_fn.parameters()).device != x.device:
            self.loss_fn = self.loss_fn.to(x.device)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.normalize:
            x = x * 2.0 - 1.0
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        self._ensure_device(pred)
        pred_p = self._prep(pred).float()
        target_p = self._prep(target).float()
        with torch.amp.autocast("cuda", enabled=False):
            out = self.loss_fn(pred_p, target_p)
        return torch.clamp_min(out, 0.0).mean()
