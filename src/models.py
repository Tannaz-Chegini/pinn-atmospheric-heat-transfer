"""Neural-network model definitions used in the PINN study.

Function/class bodies below are copied verbatim from the executed study notebook
03_oklahoma_10pct_sparse.ipynb. Shared core definitions were verified to be
identical across all seven study notebooks where applicable.

These modules are provided as readable source references. The executed notebooks
remain the authoritative experiment entry points because experiment-specific
constants and data arrays are defined there.
"""

import numpy as np
import torch
import torch.nn as nn



class ExactInitialField(nn.Module):
    # Frozen bilinear interpolation of a supplied 3-level background grid.
    def __init__(self, level_grids):
        super().__init__()
        self.register_buffer("x_axis", torch.tensor(x_km, dtype=torch.float32))
        self.register_buffer("y_axis", torch.tensor(y_km, dtype=torch.float32))
        self.register_buffer("grids", torch.tensor(level_grids, dtype=torch.float32))

    def forward(self, xy):
        xq = torch.clamp(xy[:, 0], self.x_axis[0], self.x_axis[-1])
        yq = torch.clamp(xy[:, 1], self.y_axis[0], self.y_axis[-1])

        ix = torch.searchsorted(self.x_axis, xq, right=True) - 1
        iy = torch.searchsorted(self.y_axis, yq, right=True) - 1

        ix = torch.clamp(ix, 0, len(self.x_axis) - 2)
        iy = torch.clamp(iy, 0, len(self.y_axis) - 2)

        x0 = self.x_axis[ix]
        x1 = self.x_axis[ix + 1]
        y0 = self.y_axis[iy]
        y1 = self.y_axis[iy + 1]

        wx = ((xq - x0) / (x1 - x0)).unsqueeze(1)
        wy = ((yq - y0) / (y1 - y0)).unsqueeze(1)

        f00 = self.grids[:, iy,     ix    ].T
        f10 = self.grids[:, iy,     ix + 1].T
        f01 = self.grids[:, iy + 1, ix    ].T
        f11 = self.grids[:, iy + 1, ix + 1].T

        return (
            (1.0 - wx) * (1.0 - wy) * f00
            + wx * (1.0 - wy) * f10
            + (1.0 - wx) * wy * f01
            + wx * wy * f11
        )



def make_mlp(in_dim, out_dim, hidden, layers):
    modules = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        modules += [nn.Linear(hidden, hidden), nn.Tanh()]
    modules += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*modules)



def lagrange_weights(p):
    p0, p1, p2 = [float(v) for v in levels]
    L0 = (p-p1)*(p-p2)/((p0-p1)*(p0-p2))
    L1 = (p-p0)*(p-p2)/((p1-p0)*(p1-p2))
    L2 = (p-p0)*(p-p1)/((p2-p0)*(p2-p1))
    return torch.stack([L0,L1,L2], dim=1)



class AnchoredCoordinateModel(nn.Module):
    def __init__(self, ic_model, delta_scale):
        super().__init__()
        self.ic_model = ic_model
        self.delta_scale = float(delta_scale)
        self.xmin, self.xmax = float(x_km[0]), float(x_km[-1])
        self.ymin, self.ymax = float(y_km[0]), float(y_km[-1])
        self.pmin, self.pmax = float(levels[0]), float(levels[-1])
        self.net = make_mlp(4, 1, HIDDEN, LAYERS)

    def initial_theta(self, coords):
        vals = self.ic_model(coords[:,:2])
        return torch.sum(vals * lagrange_weights(coords[:,2]), dim=1)

    def norm_coords(self, coords):
        x = 2.0*(coords[:,0]-self.xmin)/(self.xmax-self.xmin)-1.0
        y = 2.0*(coords[:,1]-self.ymin)/(self.ymax-self.ymin)-1.0
        p = 2.0*(coords[:,2]-self.pmin)/(self.pmax-self.pmin)-1.0
        tau = coords[:,3] / HISTORY_HOURS
        return torch.stack([x,y,p,tau], dim=1)

    def forward(self, coords):
        base = self.initial_theta(coords)
        corr = self.net(self.norm_coords(coords)).squeeze(1)
        return base + (coords[:,3]/HISTORY_HOURS) * self.delta_scale * corr



class AnchoredForcingModel(nn.Module):
    def __init__(self, ic_model, delta_scale, forcing_mean, forcing_std):
        super().__init__()
        self.ic_model = ic_model
        self.delta_scale = float(delta_scale)
        self.xmin, self.xmax = float(x_km[0]), float(x_km[-1])
        self.ymin, self.ymax = float(y_km[0]), float(y_km[-1])
        self.pmin, self.pmax = float(levels[0]), float(levels[-1])
        self.register_buffer("fmean", torch.tensor(forcing_mean, dtype=torch.float32))
        self.register_buffer("fstd", torch.tensor(forcing_std, dtype=torch.float32))
        self.net = make_mlp(4+len(forcing_mean), 1, HIDDEN, LAYERS)

    def initial_theta(self, coords):
        vals = self.ic_model(coords[:,:2])
        return torch.sum(vals * lagrange_weights(coords[:,2]), dim=1)

    def norm_coords(self, coords):
        x = 2.0*(coords[:,0]-self.xmin)/(self.xmax-self.xmin)-1.0
        y = 2.0*(coords[:,1]-self.ymin)/(self.ymax-self.ymin)-1.0
        p = 2.0*(coords[:,2]-self.pmin)/(self.pmax-self.pmin)-1.0
        tau = coords[:,3] / HISTORY_HOURS
        return torch.stack([x,y,p,tau], dim=1)

    def forward(self, coords, extra):
        base = self.initial_theta(coords)
        z = torch.cat(
            [self.norm_coords(coords), (extra-self.fmean)/self.fstd],
            dim=1
        )
        corr = self.net(z).squeeze(1)
        return base + (coords[:,3]/HISTORY_HOURS) * self.delta_scale * corr
