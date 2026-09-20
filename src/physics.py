"""Physics-closure and thermodynamic PDE-residual definitions used in the PINN study.

Function/class bodies below are copied verbatim from the executed study notebook
03_oklahoma_10pct_sparse.ipynb. Shared core definitions were verified to be
identical across all seven study notebooks where applicable.

These modules are provided as readable source references. The executed notebooks
remain the authoritative experiment entry points because experiment-specific
constants and data arrays are defined there.
"""

import torch
import torch.nn as nn



class HeatingClosure(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_latent = nn.Parameter(torch.tensor(0.0))  # sigmoid -> alpha=1 initially
        self.raw_b0 = nn.Parameter(torch.tensor(0.0))
        self.raw_bp = nn.Parameter(torch.tensor(0.0))
        self.raw_bs = nn.Parameter(torch.tensor(0.0))
        self.raw_bc = nn.Parameter(torch.tensor(0.0))

    def coefficients(self):
        alpha = MAX_LATENT_SCALE * torch.sigmoid(self.raw_latent)
        b0 = MAX_B0_K_PER_H * torch.tanh(self.raw_b0)
        bp = MAX_BP_K_PER_H * torch.tanh(self.raw_bp)
        bs = MAX_DIURNAL_K_PER_H * torch.tanh(self.raw_bs)
        bc = MAX_DIURNAL_K_PER_H * torch.tanh(self.raw_bc)
        return alpha,b0,bp,bs,bc

    def source(self, p, latent, sin_t, cos_t):
        alpha,b0,bp,bs,bc = self.coefficients()
        p_star = (p-850.0)/150.0
        return alpha*latent + b0 + bp*p_star + bs*sin_t + bc*cos_t

    def regularization(self):
        _,b0,bp,bs,bc = self.coefficients()
        return (
            (b0/MAX_B0_K_PER_H)**2
            + (bp/MAX_BP_K_PER_H)**2
            + (bs/MAX_DIURNAL_K_PER_H)**2
            + (bc/MAX_DIURNAL_K_PER_H)**2
        )



def pool_to_tensors(pool):
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k,v in pool.items()}



def sample_pool_batch(pool_t, batch_size, tau_max=None):
    coords = pool_t["coords"]
    if tau_max is None:
        eligible = torch.arange(len(coords), device=device)
    else:
        eligible = torch.where(coords[:,3] <= float(tau_max)+1e-7)[0]
    if len(eligible) == 0:
        raise RuntimeError("No eligible pool points.")
    chosen = eligible[torch.randint(0, len(eligible), (min(batch_size, len(eligible)),), device=device)]
    return {k: v[chosen] for k,v in pool_t.items()}



def pde_residual(model, closure, batch):
    # coords are physical x[km], y[km], p[hPa], tau[h]
    coords = batch["coords"].clone().detach().requires_grad_(True)
    theta = model(coords)

    grads = torch.autograd.grad(
        theta, coords,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
    )[0]

    source = closure.source(
        coords[:,2], batch["latent"], batch["sin"], batch["cos"]
    )

    residual = (
        grads[:,3]
        + batch["u"] * grads[:,0]
        + batch["v"] * grads[:,1]
        + batch["w"] * grads[:,2]
        - source
    )
    return residual
