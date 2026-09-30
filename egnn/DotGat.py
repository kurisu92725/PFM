import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from egnn.Dot_inter_utils import get_refine_net_inter
from egnn.DotUtils_lite import GaussianSmearing, MLP


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class DotPre_FM(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super(DotPre_FM, self).__init__()
        self.config= config
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        self.guidance_conditioning = bool(getattr(config, 'guidance_conditioning', False))
        self.predict_affinity = bool(getattr(config, 'predict_affinity', False))
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
                if self.guidance_conditioning:
                    self.drop_prob = float(getattr(config, 'condition_dropout', 0.2))
                    self.embed = nn.Sequential(
                        GaussianSmearing(
                            start=-0.05,
                            stop=1.05,
                            num_gaussians=32,
                            fixed_offset=False,
                        ),
                        nn.Linear(32, 128),
                        nn.SiLU(),
                        nn.Linear(128, emb_dim),
                    )
                    self.null = nn.Parameter(torch.zeros(emb_dim))
                    self.mask_embed = nn.Embedding(2, emb_dim)

            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net_inter=get_refine_net_inter(self.refine_net_type, config)

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 12),
        )

        if self.refine_net_type=='uni_o2_dynamic':
            self.bond_inference = nn.Sequential(
                nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim*2, 5),
            )
            if self.predict_affinity:
                self.vina_predictor = MLP(self.hidden_dim, 1, self.hidden_dim)




    def forward(self, t, xh, l_xh, b_l, p_xh, b_p, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, b_l, p_xh, b_p, node_mask, edge_mask, context, reverse,pre_train):
        def fwd(time, state):
            return self._forward(time, state, b_l, p_xh, b_p, node_mask, edge_mask, context, reverse,pre_train)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, l_xh, b_l, p_xh, b_p, node_mask, edge_mask, context, reverse=False,pre_train=False): ##node mask [0,1,2] 2随机，1有条件，0无条件
        l_h = l_xh[:, 3:].clone()
        p_h = p_xh[:, 3:].clone()
        l_x = l_xh[:, 0:3].clone()
        p_x = p_xh[:, 0:3].clone()

        if self.time_emb_dim > 0:
            if t.shape == torch.Size([]):
                h_time_l = torch.empty_like(l_h[:, 0:1]).fill_(t)
            else:
                h_time_l = t[b_l].unsqueeze(1)

            if self.time_emb_mode == 'simple':
                h_l_initial = torch.cat([l_h, h_time_l], -1)
            else:
                raise NotImplementedError
        else:
            h_l_initial= l_h

        h_protein = self.protein_atom_emb(p_h)
        init_ligand_h = self.ligand_atom_emb(h_l_initial)
        if node_mask is not None:
            if not self.guidance_conditioning:
                raise ValueError('This checkpoint does not use guidance conditioning.')
            if node_mask ==2:
                if self.drop_prob > 0.0:
                    m = (torch.rand(b_l.max().item()+1, 1, device=init_ligand_h.device) > self.drop_prob).float()  # 1 或 0
                else:
                    m = torch.ones(b_l.max().item()+1, 1, device=init_ligand_h.device)
            elif node_mask ==1:
                m = torch.ones(b_l.max().item()+1, 1, device=init_ligand_h.device)
            else:
                m = torch.zeros(b_l.max().item()+1, 1, device=init_ligand_h.device)

            e_y = self.embed(context)  # [B, emb_dim]
            e_used = m * e_y + (1.0 - m) * self.null  # [B, emb_dim]
            e_used = e_used + self.mask_embed(m.long().squeeze(-1))
            init_ligand_h = init_ligand_h + e_used[b_l]

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        h_ctx, pos_ctx, batch_ctx, mask_ligand = self.compose_context_no_rank(h_protein, init_ligand_h, p_x, l_x, b_p, b_l)

        outputs_x1= self.refine_net_inter(h_ctx, pos_ctx, mask_ligand,batch_ctx,p_x, l_x, b_p, b_l,pre_train=pre_train)
        final_pos_x1, final_h_x1 = outputs_x1['x'], outputs_x1['h']

        final_ligand_pos_x1, final_ligand_h_x1 = final_pos_x1[mask_ligand], final_h_x1[mask_ligand]
        final_h_x1 = self.v_inference(final_ligand_h_x1)


        if self.refine_net_type=='uni_o2_dynamic':
            final_score = None
            if self.predict_affinity:
                score_pre = torch.sigmoid(
                    self.vina_predictor(final_ligand_h_x1).squeeze()
                )
                final_score = scatter(score_pre, b_l, dim=0, reduce='mean')
            if edge_mask is None:
                result = torch.cat([final_ligand_pos_x1, final_h_x1], dim=-1)
                if self.predict_affinity:
                    return result, b_l, final_score
                return result, b_l

            final_node_s_f=final_ligand_h_x1[edge_mask[0]]
            final_node_t_f=final_ligand_h_x1[edge_mask[1]]
            final_edge_f=torch.cat([final_node_s_f,final_node_t_f],dim=-1)
            final_edge_f=self.bond_inference(final_edge_f)
            result = torch.cat([final_ligand_pos_x1, final_h_x1], dim=-1)
            if self.predict_affinity:
                return result, final_edge_f, final_score
            return result, final_edge_f



        if not reverse:
            return torch.cat([final_ligand_pos_x1, final_h_x1], dim=-1) ,b_l
        else:
            return torch.cat([final_ligand_pos_x1, final_h_x1], dim=-1)

    def compose_context(self,h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
        batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
        sort_idx = torch.sort(batch_ctx, stable=True).indices

        mask_ligand = torch.cat([
            torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
            torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
        ], dim=0)[sort_idx]

        batch_ctx = batch_ctx[sort_idx]
        h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]
        pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]

        return h_ctx, pos_ctx, batch_ctx, mask_ligand

    def compose_context_no_rank(self, h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
        batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
        mask_ligand = torch.cat([
            torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
            torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
        ], dim=0)

        h_ctx = torch.cat([h_protein, h_ligand], dim=0)
        pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)

        return h_ctx, pos_ctx, batch_ctx, mask_ligand
