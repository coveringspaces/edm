# training/koopman.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sigma_to_t(sigma, sigma_data: float, t_epsilon: float = 1e-4):
    # sigma = sigma_data * tan(t)
    t = torch.atan(sigma / sigma_data)
    return t.clamp(min=t_epsilon, max=0.5 * math.pi - t_epsilon)

def trigflow_alpha_s(t):
    alpha = torch.cos(t)
    s = torch.sin(t)
    return alpha, s

def cfm_dxdt_from_net(net, x, sigma, labels=None, sigma_data: float = 0.5, t_epsilon: float = 1e-4):
    """
    net(x, sigma) returns x0_hat.
    Returns dx/dt = (alpha*x - x0_hat)/s  where alpha=cos(t), s=sin(t), t=atan(sigma/sigma_data).
    """
    x0_hat = net(x, sigma, labels)

    t = sigma_to_t(sigma.reshape(-1, 1, 1, 1), sigma_data=sigma_data, t_epsilon=t_epsilon)
    alpha, s = trigflow_alpha_s(t)
    dxdt = (alpha * x - x0_hat) / s.clamp(min=1e-6)
    return dxdt

class PsiNetComplex(nn.Module):
    """
    Outputs psi(z) in C^k as two real tensors: [B,k,2] = (real, imag).
    """
    def __init__(self, img_channels=3, img_resolution=32, k=16):
        super().__init__()
        self.k = k
        C = img_channels
        self.conv = nn.Sequential(
            nn.Conv2d(C, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 1, 256), nn.SiLU(),
            nn.Linear(256, 2 * k),  # real+imag
        )

    def forward(self, t_scalar, x):
        # t_scalar: [B,1], x: [B,C,H,W]
        h = self.conv(x).mean(dim=(2,3))          # [B,256]
        h = torch.cat([t_scalar, h], dim=1)       # [B,257]
        out = self.head(h)                        # [B,2k]
        B = x.shape[0]
        out = out.view(B, self.k, 2)              # [B,k,2]
        return out

def apply_generator_to_psi(psi, t_scalar, x, dxdt):
    """
    psi: [B,k,2] (real/imag)
    returns Lpsi: [B,k,2]
    """
    B, k, _ = psi.shape
    Lpsi = torch.zeros_like(psi)

    # compute per-(i,part) so it's correct (start with small k for speed)
    for i in range(k):
        for part in range(2):  # 0 real, 1 imag
            out = psi[:, i, part]  # [B]

            d_out_dt = torch.autograd.grad(
                outputs=out, inputs=t_scalar,
                grad_outputs=torch.ones_like(out),
                create_graph=True, retain_graph=True
            )[0].squeeze(1)  # [B]

            grad_x = torch.autograd.grad(
                outputs=out, inputs=x,
                grad_outputs=torch.ones_like(out),
                create_graph=True, retain_graph=True
            )[0]  # [B,C,H,W]

            jvp = (grad_x * dxdt).reshape(B, -1).sum(dim=1)  # [B]

            Lpsi[:, i, part] = d_out_dt + jvp

    return Lpsi

def inner_prod(a, b):
    """
    a,b: [B,k,2] complex vectors (real/imag)
    returns matrix G: [k,k,2] representing complex Gram:
      G_ij = <a_i, b_j> approx mean(conj(a_i)*b_j)
    We'll return real/imag parts.
    """
    # a_i = a[:,i,:], b_j = b[:,j,:]
    # conj(a) * b = (ar - i ai)(br + i bi)
    # = (ar*br + ai*bi) + i(ar*bi - ai*br)
    ar, ai = a[..., 0], a[..., 1]  # [B,k]
    br, bi = b[..., 0], b[..., 1]  # [B,k]

    # compute <a_i, b_j> for all i,j
    # real: mean_b (ar_i*br_j + ai_i*bi_j)
    real = (ar.transpose(0,1) @ br + ai.transpose(0,1) @ bi) / ar.shape[0]  # [k,k]
    imag = (ar.transpose(0,1) @ bi - ai.transpose(0,1) @ br) / ar.shape[0]  # [k,k]
    return torch.stack([real, imag], dim=-1)  # [k,k,2]

class NormalKoopmanObjective(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(k))  # phases

    def forward(self, psi, Lpsi):
        """
        psi, Lpsi: [B,k,2]
        returns scalar loss matching Eq (4)
        """
        k = psi.shape[1]
        # A_ii = <psi_i, L psi_i> (complex)
        A = inner_prod(psi, Lpsi)  # [k,k,2]
        # G_ij = <psi_i, psi_j>
        G = inner_prod(psi, psi)   # [k,k,2]

        # extract diagonal A_ii
        A_diag = A[torch.arange(k), torch.arange(k)]  # [k,2] real/imag

        # compute Re( e^{i phi_i} A_ii )
        # e^{i phi} = cos + i sin
        c = torch.cos(self.phi)
        s = torch.sin(self.phi)
        # Re( (c + i s)(a + i b) ) = c*a - s*b
        align = c * A_diag[:, 0] - s * A_diag[:, 1]  # [k]
        term1 = -2.0 * align.sum()

        # term2 = sum_{ij} cos(phi_i - phi_j) * |G_ij|^2
        # |G_ij|^2 = real^2 + imag^2
        abs2 = (G[..., 0] ** 2 + G[..., 1] ** 2)  # [k,k]
        phi_diff = self.phi[:, None] - self.phi[None, :]
        term2 = (torch.cos(phi_diff) * abs2).sum()

        loss = term1 + term2
        return loss

def sample_z_from_data(y, sigma_data, P_mean, P_std, sigma_min, sigma_max, t_eps):
    B = y.shape[0]
    device = y.device
    rnd = torch.randn([B,1,1,1], device=device)
    sigma = (rnd * P_std + P_mean).exp().clamp(min=sigma_min, max=sigma_max)

    t = sigma_to_t(sigma, sigma_data=sigma_data, t_epsilon=t_eps)   # [B,1,1,1]
    alpha, s = trigflow_alpha_s(t)
    eps = torch.randn_like(y)
    x = alpha * y + s * sigma_data * eps

    return x, sigma, t

def normal_koopman_train_step(psi_net, obj, cfm_net, y, labels=None,
                              sigma_data=0.5, P_mean=-1.2, P_std=1.2,
                              sigma_min=1e-3, sigma_max=80, t_eps=1e-4):
    # sample state
    x, sigma, t = sample_z_from_data(y, sigma_data, P_mean, P_std, sigma_min, sigma_max, t_eps)

    B = y.shape[0]
    t_scalar = t.reshape(B,1).detach().requires_grad_(True)
    x = x.detach().requires_grad_(True)

    # generator f = dx/dt from frozen CFM net
    with torch.no_grad():
        dxdt = cfm_dxdt_from_net(cfm_net, x, sigma, labels, sigma_data=sigma_data, t_epsilon=t_eps)

    # psi and L psi
    psi = psi_net(t_scalar, x)                       # [B,k,2]
    Lpsi = apply_generator_to_psi(psi, t_scalar, x, dxdt)  # [B,k,2]

    # objective Eq (4)
    loss = obj(psi, Lpsi)
    return loss






