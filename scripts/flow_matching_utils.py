import torch
from torch import nn

from scripts.traj_sec import SHA_POS_Sampler


class CFM_CD_PERTURB(nn.Module):
    def __init__(self, sigma=0.0):
        super().__init__()
        self.x_perturb = nn.Parameter(torch.tensor(1.5))
        self.h_perturb = nn.Parameter(torch.tensor(1.5))
        self.sigma = sigma

    def forward(self, x0, x1, x_bar, b_l, t=None):
        if t is None:
            t = torch.rand(b_l.max().item() + 1, device=x0.device)
        time = t[b_l].unsqueeze(1)
        perturb = torch.cat([
            x_bar[:, :3] * self.x_perturb,
            x_bar[:, 3:] * self.h_perturb,
        ], dim=-1)
        mean = (1 - time) * x0 + time * x1 + time * (1 - time) * perturb
        state = mean + self.sigma * torch.randn_like(x0)
        velocity = x1 - x0 + (1 - 2 * time) * perturb
        return t, state, velocity


class POS_HA:
    def __init__(self, sigma=0.0):
        self.sampler = SHA_POS_Sampler()

    def sample_location_and_conditional_flow(self, x0, x1, b_l, t=None):
        x0, _ = self.sampler.sample_plan(x0, x1, b_l)
        return x0
