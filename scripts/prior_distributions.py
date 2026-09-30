import torch
from utils.remove_mean import sample_ligand_pos_crossdock,sample_ligand_v_crossdock##flows.utils

class PositionFeaturePrior_PFM(torch.nn.Module):
    def __init__(self, n_dim, in_node_nf):
        super().__init__()
        self.n_dim = n_dim
        self.in_node_nf = in_node_nf

    def forward(self, z_x, z_h,batch):
        return None
    def sample(self, ligand_num_atoms,device):
        total_num_atoms = torch.tensor(ligand_num_atoms,dtype=torch.long).sum().to(device)
        v_size=self.in_node_nf
        z_x = sample_ligand_pos_crossdock(total_num_atoms,device)
        z_h = sample_ligand_v_crossdock(total_num_atoms,v_size,device)
        return z_x, z_h

