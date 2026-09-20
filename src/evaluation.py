"""Common evaluation, prediction, and diagnostic helpers used in all study notebooks.

Function/class bodies below are copied verbatim from the executed study notebook
03_oklahoma_10pct_sparse.ipynb. Shared core definitions were verified to be
identical across all seven study notebooks where applicable.

These modules are provided as readable source references. The executed notebooks
remain the authoritative experiment entry points because experiment-specific
constants and data arrays are defined there.
"""

import numpy as np
import torch



def theta_to_T(theta, p_hpa):
    return theta / ((P0_HPA/p_hpa)**KAPPA)



def rmse(a,b):
    a,b = np.asarray(a,float), np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2)))



def mae(a,b):
    a,b = np.asarray(a,float), np.asarray(b,float)
    return float(np.mean(np.abs(a-b)))



def corr(a,b):
    a,b = np.asarray(a,float), np.asarray(b,float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])



def predict_coord(model, coords):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0,len(coords),8192):
            c = torch.tensor(coords[s:s+8192], dtype=torch.float32, device=device)
            out.append(model(c).cpu().numpy())
    return np.concatenate(out)



def predict_forcing(model, coords, extra):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0,len(coords),8192):
            c = torch.tensor(coords[s:s+8192], dtype=torch.float32, device=device)
            e = torch.tensor(extra[s:s+8192], dtype=torch.float32, device=device)
            out.append(model(c,e).cpu().numpy())
    return np.concatenate(out)



def future_physics_rms(model, closure, bundle, seed=999):
    pool_t = pool_to_tensors(bundle.future_phys)
    rng = np.random.default_rng(seed)
    n = min(12000, len(bundle.future_phys["coords"]))
    idx_np = rng.choice(len(bundle.future_phys["coords"]), size=n, replace=False)
    idx = torch.tensor(idx_np, dtype=torch.long, device=device)
    sub = {k:v[idx] for k,v in pool_t.items()}
    res = pde_residual(model, closure, sub)
    return float(torch.sqrt(torch.mean(res**2)).detach().cpu())
