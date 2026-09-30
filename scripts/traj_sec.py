import torch
from scipy.optimize import linear_sum_assignment


class SHA_POS_Sampler:
    def sample_plan(self, x0, x1, b_l, pre_train=False):
        aligned = []
        for index in range(b_l.max().item() + 1):
            mask = b_l == index
            source = x0[mask, :3]
            target = x1[mask, :3]
            cost = (target[:, None] - source[None]).square().sum(dim=-1)
            _, columns = linear_sum_assignment(cost.detach().cpu().numpy())
            aligned.append(x0[mask][torch.as_tensor(columns, device=x0.device)])
        return torch.cat(aligned, dim=0), x1
