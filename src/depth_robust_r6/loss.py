"""Training objectives. The measured delta is always supervised; every auxiliary term is additive.

    supervised      MSE(student delta, measured delta)                       -- every arm
    response        MSE(student delta, stopgrad teacher delta)               -- teacher arms (B01, B11)
    latent          1 - cos(proj(student z), stopgrad teacher z)             -- teacher arms
    reconstruction  MSE(decoded per-cell vector, the cell's own full-observation expression,
                        gene-standardised)                                   -- denoise-control only

The teacher sees the same cells at full observation and the same drugs; it is a fixed network
trained on the same fitting drugs. Its output is detached here, again, on purpose: the teacher's
output must never receive a gradient and never replaces the measured target.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised(pred, y):
    return F.mse_loss(pred, y)


def response_consistency(pred_student, pred_teacher):
    return F.mse_loss(pred_student, pred_teacher.detach())


def latent_consistency(z_student_proj, z_teacher):
    return (1.0 - F.cosine_similarity(z_student_proj, z_teacher.detach(), dim=-1)).mean()


def reconstruction(decoded, x_full, x_mean, x_sd):
    return F.mse_loss(decoded, ((x_full - x_mean) / x_sd).detach())


def total(parts: dict, weights: dict) -> torch.Tensor:
    return sum(weights.get(k, 0.0) * v for k, v in parts.items() if weights.get(k, 0.0) != 0.0)
