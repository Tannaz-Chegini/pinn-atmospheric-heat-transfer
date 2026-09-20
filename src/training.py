"""Training routines used for the neural-network baselines and PINN.

Function/class bodies below are copied verbatim from the executed study notebook
03_oklahoma_10pct_sparse.ipynb. Shared core definitions were verified to be
identical across all seven study notebooks where applicable.

These modules are provided as readable source references. The executed notebooks
remain the authoritative experiment entry points because experiment-specific
constants and data arrays are defined there.
"""

import numpy as np
import torch



def random_idx(n, batch_size):
    return torch.randint(0, n, (min(batch_size,n),), device=device)



def train_coordinate_baseline(bundle, ic_model, seed):
    set_seed(seed)
    model = AnchoredCoordinateModel(ic_model, bundle.data_scale).to(device)
    coords = torch.tensor(bundle.train_coords, dtype=torch.float32, device=device)
    target = torch.tensor(bundle.train_theta, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TOTAL_MAIN_STEPS, eta_min=1e-5
    )

    for _ in range(TOTAL_MAIN_STEPS):
        idx = random_idx(len(coords), DATA_BATCH)
        pred = model(coords[idx])
        loss = torch.mean(((pred-target[idx])/bundle.data_scale)**2)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sch.step()

    return model



def train_forcing_baseline(bundle, ic_model, seed):
    set_seed(seed + 10000)
    model = AnchoredForcingModel(
        ic_model, bundle.data_scale, bundle.forcing_mean, bundle.forcing_std
    ).to(device)

    coords = torch.tensor(bundle.train_coords, dtype=torch.float32, device=device)
    target = torch.tensor(bundle.train_theta, dtype=torch.float32, device=device)
    extra = torch.tensor(bundle.train_forcing_extra, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TOTAL_MAIN_STEPS, eta_min=1e-5
    )

    for _ in range(TOTAL_MAIN_STEPS):
        idx = random_idx(len(coords), DATA_BATCH)
        pred = model(coords[idx], extra[idx])
        loss = torch.mean(((pred-target[idx])/bundle.data_scale)**2)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sch.step()

    return model



def closure_values(closure):
    with torch.no_grad():
        return [float(v.detach().cpu()) for v in closure.coefficients()]



def train_pinn(bundle, ic_model, seed):
    set_seed(seed)

    model = AnchoredCoordinateModel(ic_model, bundle.data_scale).to(device)
    closure = HeatingClosure().to(device)

    coords = torch.tensor(bundle.train_coords, dtype=torch.float32, device=device)
    target = torch.tensor(bundle.train_theta, dtype=torch.float32, device=device)

    past_pool = pool_to_tensors(bundle.past_phys)
    future_pool = pool_to_tensors(bundle.future_phys)
    boundary_pool = pool_to_tensors(bundle.future_boundary)

    # Stage A: learn closure from past only.
    params = list(model.parameters()) + list(closure.parameters())
    opt = torch.optim.Adam(params, lr=LEARNING_RATE)

    for step in range(STAGE_A_STEPS):
        idx = random_idx(len(coords), DATA_BATCH)

        pred = model(coords[idx])
        data_loss = torch.mean(((pred-target[idx])/bundle.data_scale)**2)

        pb = sample_pool_batch(past_pool, PHYS_BATCH)
        res = pde_residual(model, closure, pb)
        phys_loss = torch.mean((res/bundle.tendency_scale)**2)

        loss = (
            data_loss
            + LAMBDA_PHYSICS*phys_loss
            + LAMBDA_SOURCE_REG*closure.regularization()
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()

        if step % 500 == 0:
            print(
                f"      Stage A {step:4d}/{STAGE_A_STEPS} "
                f"data={data_loss.item():.4f} physics={phys_loss.item():.4f}"
            )

    frozen_closure = closure_values(closure)
    for par in closure.parameters():
        par.requires_grad_(False)

    # Future causal stages.
    for horizon in range(1, FORECAST_HOURS+1):
        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*0.5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=FUTURE_STAGE_STEPS, eta_min=1e-5
        )

        for _ in range(FUTURE_STAGE_STEPS):
            idx = random_idx(len(coords), DATA_BATCH)
            pred = model(coords[idx])
            data_loss = torch.mean(((pred-target[idx])/bundle.data_scale)**2)

            pb0 = sample_pool_batch(past_pool, PHYS_BATCH//2)
            pbf = sample_pool_batch(future_pool, PHYS_BATCH//2, tau_max=horizon)

            r0 = pde_residual(model, closure, pb0)
            rf = pde_residual(model, closure, pbf)
            phys_loss = 0.5*(
                torch.mean((r0/bundle.tendency_scale)**2)
                + torch.mean((rf/bundle.tendency_scale)**2)
            )

            bb = sample_pool_batch(boundary_pool, BOUNDARY_BATCH, tau_max=horizon)
            theta_b = model(bb["coords"])
            theta_0 = model.initial_theta(bb["coords"])
            boundary_loss = torch.mean(((theta_b-theta_0)/bundle.data_scale)**2)

            loss = (
                data_loss
                + LAMBDA_PHYSICS*phys_loss
                + LAMBDA_BOUNDARY*boundary_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sch.step()

        print(
            f"      Causal +{horizon}h complete | "
            f"data={data_loss.item():.4f}, physics={phys_loss.item():.4f}, "
            f"boundary={boundary_loss.item():.4f}"
        )

    # Closure must be bitwise unchanged after future stages.
    assert np.allclose(frozen_closure, closure_values(closure), atol=0, rtol=0)

    return model, closure
