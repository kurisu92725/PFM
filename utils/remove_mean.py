import torch
from torch_geometric.utils import add_self_loops, scatter
def remove_mean_crossdock(px,lx,p_batch,l_batch,pre_train=False):
    p_mean=scatter(px,p_batch,dim=0,reduce='mean')
    L_mean=scatter(lx,l_batch,dim=0,reduce='mean')
    p_mean_ = p_mean[p_batch]
    if pre_train:
        l_mean_ = L_mean[l_batch]
    else:
        l_mean_ = p_mean[l_batch]

    return px-p_mean_,lx-l_mean_

def sample_ligand_pos_crossdock(total_num,device):
    x = torch.randn((total_num,3),device=device)
    return x


def sample_ligand_v_crossdock(total_num, v_size,device):
    x = torch.randn((total_num,v_size),device=device)
    return x