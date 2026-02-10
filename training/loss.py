# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Simple Gaussian-path Conditional Flow Matching (CFM) loss.
# Trains velocity v so that x = y + sigma*eps has dx/dsigma = eps.

@persistence.persistent_class
class CFMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, sigma_min=1e-3, sigma_max=80, t_epsilon=1e-4):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.t_epsilon = t_epsilon

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # Optional augmentation (match EDMLoss).
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        # Sample sigma ~ lognormal (EDM-style).
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)

        # Convert sigma -> t for TrigFlow.
        t = torch.atan(sigma / self.sigma_data)
        # Optional safety so sin/cos never hit exact 0/1 due to numerical edge cases.
        t = t.clamp(min=self.t_epsilon, max=0.5 * torch.pi - self.t_epsilon)

        alpha = torch.cos(t)
        s = torch.sin(t)

        # TrigFlow forward path.
        eps = torch.randn_like(y)
        x = alpha * y + s * self.sigma_data * eps

        # Net returns x0_hat.
        x0_hat, logvar = net(x, t, labels, augment_labels=augment_labels, return_logvar=True)

        # Recover v_pred from x0_hat:
        # x0_hat = alpha*x - s*sigma_data*v  =>  v = (alpha*x - x0_hat)/(s*sigma_data)
        denom = (s * self.sigma_data).clamp(min=1e-8)
        v_pred = (alpha * x - x0_hat) / denom

        # TrigFlow target velocity.
        v_star = alpha * eps - s * (y / self.sigma_data)

        return (1/logvar.exp()) * (v_pred - v_star) ** 2 + logvar

