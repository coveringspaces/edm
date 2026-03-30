# Koopman Loss Documentation

This document describes the loss function implemented in `training/koopman.py`,
how data flows through each stage of `KoopmanLoss.__call__`, and how the
`NestedLoRALossFunctionEVD` custom autograd function computes gradients.

---

## 1. Background: NestedLoRA Objective

We approximate the time-augmented Koopman generator

$$\tilde{\mathcal{L}} \;\approx\; \sum_{i=1}^{k} e^{i\varphi_i}\,\psi_{\theta,i}\,\psi_{\theta,i}^\dagger$$

by minimising the Hilbert-Schmidt residual (Eq. 6 from the writeup):

$$\mathcal{L} = -2\sum_{i=1}^{k} \operatorname{Re}\!\bigl(e^{i\varphi_i}\langle\psi_{\theta,i},\,\tilde{\mathcal{L}}\psi_{\theta,i}\rangle\bigr) + \sum_{i,j=1}^{k}\cos(\varphi_i - \varphi_j)\,|\langle\psi_{\theta,i},\,\psi_{\theta,j}\rangle|^2$$

The first sum is the **operator term** (drives eigenfunctions toward the true
Koopman eigenfunctions). The double sum is the **metric term** (enforces
orthogonality between eigenfunctions and, together with the nesting structure,
ensures they are found in order of decreasing eigenvalue magnitude).

### Sequential nesting

The metric term uses an upper-triangular structure mask
$M_{i,\ell} = \mathbf{1}[i \le \ell]$ (from `get_sequential_nesting_masks`).
This implements the "sequential nesting" from the Neural SVD / NestedLoRA
paper, which recovers eigenfunctions in order: the level-$\ell$ sub-problem
involves only the first $\ell$ eigenfunctions.

---

## 2. Learnable Parameters

There are three trainable modules:

| Module | Parameters | Role |
|---|---|---|
| `KoopmanEigenNet` (psi_net) | Neural network weights | Eigenfunction *shape*: $\psi_{\theta,i}(\tilde{x})$ |
| `KoopmanPhases` (phase_net) | $\varphi_i \in \mathbb{R}^k$ | Eigenvalue *phase*: $e^{i\varphi_i}$ |
| `KoopmanMagnitudes` (mag_net) | $\log r_i \in \mathbb{R}^k$ | Eigenvalue *magnitude*: $r_i = e^{\log r_i} > 0$ |

At the optimum, the full eigenvalue is $\lambda_i = r_i^2 \cdot e^{i\varphi_i}$.

### Why decouple magnitude from norm?

Previously, the eigenvalue magnitude was $|\lambda_i| = \|\psi_{\theta,i}\|^2$
(the batch-averaged squared norm of the network output). This entangles
eigenvalue magnitude with network output scale, leading to unbounded growth:
the operator term drives $\|\psi\|$ larger while the quartic metric term
provides the only counter-pressure, creating a fragile balance.

Now, $\psi$ is **normalised** to unit norm (detached) in the loss, and the
magnitude information lives in the $k$ scalar parameters $r_i$. The network
only learns eigenfunction *directions*, and the $r_i$ scalars find stable
equilibria via the quadratic/quartic balance in the loss.

---

## 3. Forward Pass: `KoopmanLoss.__call__`

### Step A — Sample $(x, t)$ on the TrigFlow curve

```
sigma ~ exp(Normal(P_mean, P_std)),  clamped to [sigma_min, sigma_max]
t     = atan(sigma / sigma_data),    clamped to [t_epsilon, pi/2 - t_epsilon]
alpha = cos(t),   s = sin(t)
x     = alpha * y + s * sigma_data * eps       (y = clean image, eps ~ N(0,I))
```

This samples a noisy image $x$ at TrigFlow time $t$.

### Step B — Frozen CFM vector field

```python
with torch.no_grad():
    xdot = cfm_dxdt_from_net(cfm_net, x, t, ...)
```

The frozen CFM teacher provides the time-augmented vector field
$\tilde{v}(\tilde{x}) = [1;\; v_t(x)]$, where $\dot{x} = v_t(x)$ is
computed as `(cos(t)*x - x0_hat) / sin(t)`. Here `tdot = 1` (time evolves
at unit rate).

### Step C — JVP: compute $(\psi, \mathcal{L}\psi)$ simultaneously

```python
psi_hat, Lpsi_hat = jvp(f, (x, t), (xdot, tdot))
```

A single forward-mode JVP through `psi_net` gives:
- `psi_hat`: the eigenfunction values $\psi_\theta(x, t)$, shape `(B, 2k)`
- `Lpsi_hat`: the Koopman generator applied to $\psi$, i.e.
  $\mathcal{L}\psi = \nabla_x\psi \cdot v_t(x) + \partial_t\psi$, shape `(B, 2k)`

The `2k` output is split into real and imaginary parts: `psi_re, psi_im` and
`Lpsi_re, Lpsi_im`, each `(B, k)`.

### Step D — Normalise and rescale

```python
# Per-mode batch norm (detached — no gradient through the norm)
norm = sqrt(mean(|psi_i|^2, dim=batch)).detach()    # (k,)
psi_hat  /= norm     # now ||psi_hat_i|| ≈ 1
Lpsi_hat /= norm     # same scale factor

# Rescale by learned magnitudes
r = mag_net(...)     # (k,), positive
psi_hat  *= r
Lpsi_hat *= r
```

After this, the tensors entering the loss are:

$$\psi_i^{\text{scaled}} = r_i \cdot \hat\psi_i, \qquad \mathcal{L}\psi_i^{\text{scaled}} = r_i \cdot \mathcal{L}\hat\psi_i$$

where $\|\hat\psi_i\| \approx 1$. Substituting into the NestedLoRA loss:

- **Operator term**: $-2\sum_i r_i^2 \operatorname{Re}(e^{i\varphi_i} \langle\hat\psi_i, \mathcal{L}\hat\psi_i\rangle) / S$
- **Metric term**: $\sum_{i,j} r_i^2 r_j^2 \cos(\varphi_i - \varphi_j) |\langle\hat\psi_i, \hat\psi_j\rangle|^2$

The equilibrium for $r_i$ is finite: operator drives $r_i$ up (quadratic),
metric pushes it down (quartic).

### Step E — Split batch for metric

The batch is split in half for the metric term (NeuralSVD f1/f2 trick):
- `psi_A = psi[:B//2]` — used to compute Gram matrix $\Lambda_A$
- `psi_B = psi[B//2:]` — used to compute Gram matrix $\Lambda_B$

Using two independent half-batch Gram estimates avoids the positive bias that
a single-batch estimator would have ($\mathbb{E}[\Lambda^2] \ne (\mathbb{E}[\Lambda])^2$).

The full batch is used for the operator term (it has no such bias issue).

### Step F — `NestedLoRALossFunctionEVD.apply`

This is the custom autograd function described in Section 4 below.

---

## 4. Custom Autograd: `NestedLoRALossFunctionEVD`

A custom `torch.autograd.Function` is used so that metric-term gradients are
computed with the correct split-batch structure, and to avoid autograd overhead
on the Gram matrix operations.

### Inputs

| Tensor | Shape | Source |
|---|---|---|
| `psi_re, psi_im` | `(B, k)` | Full batch, normalised+rescaled |
| `Lpsi_re, Lpsi_im` | `(B, k)` | JVP output, normalised+rescaled |
| `psi_A_re, psi_A_im` | `(B/2, k)` | First half of `psi` |
| `psi_B_re, psi_B_im` | `(B/2, k)` | Second half of `psi` |
| `cos_phi, sin_phi` | `(k,)` | From `KoopmanPhases` |
| `operator_scale` | scalar | Divides operator term |
| `struct_mask` | `(k, k)` | Upper-triangular nesting mask |

### Forward

1. **Gram matrices** (metric term, half-batches):
   ```
   Lambda_A[i,j] = psi_A_i^dagger @ psi_A_j / (B/2)     complex (k,k)
   Lambda_B[i,j] = psi_B_i^dagger @ psi_B_j / (B/2)     complex (k,k)
   ```

2. **Phase-weighted mask**:
   ```
   matrix_mask[i,j] = struct_mask[i,j] * cos(phi_i - phi_j)
   ```

3. **Metric loss**:
   ```
   D[i,j]      = Re(conj(Lambda_B[i,j]) * Lambda_A[i,j])
   loss_metric  = sum over (i,j) of matrix_mask[i,j] * D[i,j]
   ```

4. **Operator inner products** (full batch):
   ```
   ip_i = <psi_i, L psi_i> / B      complex, (k,)
   ```

5. **Operator loss**:
   ```
   loss_operator = (-2 / scale) * sum_i Re(e^{i phi_i} * ip_i)
   ```

6. **Total**: `loss = loss_operator + loss_metric`

### Backward

Gradients are computed analytically for each input group:

**Metric gradients** (half-batches only):
```
d loss / d psi_A = (2/B_half) * einsum(matrix_mask * Lambda_B, psi_A)
d loss / d psi_B = (2/B_half) * einsum(matrix_mask * Lambda_A, psi_B)
```
(with appropriate complex conjugation for re/im parts)

**Operator gradients** (full batch only):
```
d loss / d psi_re  = (-2 / (scale * B)) * (cos_phi * Lpsi_re - sin_phi * Lpsi_im)
d loss / d Lpsi_re = (-2 / (scale * B)) * (cos_phi * psi_re  + sin_phi * psi_im)
```
(and analogously for `_im` parts)

The `Lpsi` gradients flow back through the JVP into `psi_net` weights.

**Phase gradients** (from both terms):
```
d loss / d cos_phi += (-2/scale) * ip_re        (from operator)
d loss / d sin_phi += ( 2/scale) * ip_im        (from operator)
d loss / d cos_phi += M_seq_D @ cos_phi + ...   (from metric)
d loss / d sin_phi += M_seq_D @ sin_phi + ...   (from metric)
```

These propagate to `phase_net.phi` through the chain rule
$\partial/\partial\varphi = -\sin\varphi \cdot \partial/\partial\cos\varphi + \cos\varphi \cdot \partial/\partial\sin\varphi$.

**Magnitude gradients**: Not inside the custom function. Since `r` multiplies
`psi` and `Lpsi` *before* they enter `NestedLoRALossFunctionEVD`, autograd
handles the chain rule automatically:
$\partial\mathcal{L}/\partial r_i = \psi_i \cdot \partial\mathcal{L}/\partial(\text{psi\_scaled}_i) + \mathcal{L}\psi_i \cdot \partial\mathcal{L}/\partial(\text{Lpsi\_scaled}_i)$.

---

## 5. Gradient Flow Diagram

```
images ──► sample (x,t) ──► frozen CFM ──► xdot
                │                              │
                └──────────┐   ┌───────────────┘
                           ▼   ▼
                    JVP through psi_net
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                 psi_hat      Lpsi_hat       (B, 2k)
                    │             │
              split re/im    split re/im
                    │             │
                    ▼             ▼
               psi_re/im    Lpsi_re/im      (B, k)
                    │             │
          ┌─── normalise (detached norm) ───┐
          │         │             │         │
          │    psi_hat_re/im  Lpsi_hat_re/im│  (unit norm)
          │         │             │         │
          │    ┌── multiply by r (from mag_net) ──┐
          │    │    │             │               │
          │    ▼    ▼             ▼               │
          │   psi_scaled     Lpsi_scaled          │
          │    │    │             │               │
          │    │  split A/B halves                │
          │    │    │    │        │               │
          │    ▼    ▼    ▼        ▼               │
          │  NestedLoRALossFunctionEVD ◄── cos_phi, sin_phi
          │              │                        │
          │              ▼                        │
          │           loss_svd                    │
          │              │                        │
          │         backward()                    │
          │         ┌────┴─────────┐              │
          │         ▼              ▼              │
          │   grad_psi_*     grad_Lpsi_*          │
          │         │              │              │
          │         │    (chain rule through r)   │
          │         ▼              ▼              │
          │    grad into       grad into          │
          │    psi_net         psi_net (via JVP)  │
          │                                       │
          └───► grad into mag_net.log_r ◄─────────┘
                grad into phase_net.phi
```

---

## 6. Inference

At inference time (see `koopman_inference.py`), the Koopman spectral
reconstruction is:

$$x_1 \approx \operatorname{Re}\biggl(\sum_{i=1}^{k} e^{\lambda_i T} \cdot c_i \cdot \psi_i(x_0, t_0)\biggr)$$

where:
- $\lambda_i = r_i^2 \cdot e^{i\varphi_i}$ (from `mag_net` and `phase_net`)
- $c_i = \langle\psi_i, \pi_{\text{coord}}\rangle$ (coefficients estimated by Monte Carlo)
- $T = \pi/2$ for the full TrigFlow trajectory

The eigenfunction output from `psi_net` is normalised by the Monte Carlo norm
estimate and rescaled by $r_i$ to match training, so coefficients and
eigenvalues are consistent.

---

## 7. Logged Statistics

| Stat name | Meaning |
|---|---|
| `Loss/koopman` | Total loss (operator + metric) |
| `Loss/loss_operator` | Operator term only |
| `Loss/loss_metric` | Metric term only |
| `Loss/psi_raw_rms` | RMS of raw network output (before normalisation) |
| `Loss/Lpsi_raw_rms` | RMS of raw JVP output (before normalisation) |
| `Loss/psi_rms` | RMS of normalised+rescaled psi (should be ~r_mean) |
| `Loss/Lpsi_rms` | RMS of normalised+rescaled Lpsi |
| `Loss/dtpsi_rms` | Time-derivative component of Lpsi |
| `Loss/dxpsi_rms` | Spatial-derivative component of Lpsi |
| `Loss/xdot_rms` | RMS of CFM vector field |
| `Loss/abs_ip_mean` | Mean magnitude of per-mode inner products |
| `Eigenvalues/r_min,max,mean` | Learned magnitude parameters |
| `Eigenvalues/psi_raw_norm` | Mean per-mode norm of raw psi (before normalisation) |
| `Eigenvalues/lam_mag_*` | Eigenvalue magnitudes $r_i^2$ |
| `Eigenvalues/lam_mag_XX` | Per-eigenfunction magnitude |
| `Grads/grad_norm` | Global gradient norm before clipping |
| `Grads/grad_norm_clipped` | Global gradient norm after clipping |
