import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_cluster import knn
from torch_geometric.utils import add_self_loops, scatter
import numpy as np
from torch_scatter import scatter_softmax, scatter_sum
from egnn.DotUtils_lite import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product


def get_refine_net_inter(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral_Interact(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'uni_o2_dynamic':
        refine_net = UniTransformerO2TwoUpdateGeneral_Interact_Dynamic(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'uni_o2_dynamic_v_s_cfg':
        refine_net = UniTransformerO2TwoUpdateGeneral_Interact_Dynamic(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )

    else:
        raise ValueError(refine_net_type)
    return refine_net


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BaseX2HAttLayer_Inter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer_Inter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral_Inter(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer_Inter(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer_Inter(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False,protein_mode=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)  #
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out

        if protein_mode:
            return x2h_out

        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)


        return x2h_out, delta_x



class UniTransformerO2TwoUpdateGeneral_Interact(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='knn', ew_net_type='global',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=False, sync_twoup=False,
                 connect_type='SE'):  ##SA,SE
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks  ##1
        self.num_layers = num_layers  ##9
        self.hidden_dim = hidden_dim  ##128
        self.n_heads = n_heads  ##1
        self.num_r_gaussian = num_r_gaussian  ##50
        self.edge_feat_dim = edge_feat_dim  ##0
        self.act_fn = act_fn  ##relu
        self.norm = norm  ##True
        self.num_node_types = num_node_types  ##8
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none] ##knn
        self.k = k
        self.connect_type = connect_type
        ###add
        self.p_k = 24
        self.l_k = 32#48
        self.pl_k = 32#48
        self.only_l = 40
        ###end add
        self.ew_net_type = ew_net_type  # [r, m, none] global

        self.num_x2h = num_x2h  #
        self.num_h2x = num_h2x  ##
        self.num_init_x2h = num_init_x2h  ##1
        self.num_init_h2x = num_init_h2x  ##0
        self.r_max = r_max  ##10
        self.x2h_out_fc = x2h_out_fc  ##True
        self.sync_twoup = sync_twoup  ##False
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)  #
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)  ##20-128

        self.init_h_emb_layer = self._build_init_h_layer()  #
        self.base_block_ligand = self._build_share_blocks()
        self.base_block_protein = self._build_share_blocks()
        self.base_block_pl = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block_ligand.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral_Inter(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):  #
            layer = AttentionLayerO2TwoUpdateNodeGeneral_Inter(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    def _connect_edge_sep(self, pos_protein, pos_ligand, batch_protein, batch_ligand):

        p_atom_num = pos_protein.size(0)

        p_edge_index = knn(pos_protein, pos_protein, self.p_k + 1, batch_protein,
                           batch_protein)  #
        l_edge_index = knn(pos_ligand, pos_ligand, self.l_k + 1, batch_ligand, batch_ligand)
        p_inner_mask = p_edge_index[0] != p_edge_index[1]
        l_inner_mask = l_edge_index[0] != l_edge_index[1]
        p_edge_index = p_edge_index[:, p_inner_mask]
        l_edge_index = l_edge_index[:, l_inner_mask] + p_atom_num

        p_l_edge_index = knn(pos_protein, pos_ligand, self.pl_k, batch_protein, batch_ligand)
        p_l_edge_target = torch.cat([p_l_edge_index[0] + p_atom_num, p_l_edge_index[1]]).unsqueeze(0)
        p_l_edge_source = torch.cat([p_l_edge_index[1], p_l_edge_index[0] + p_atom_num]).unsqueeze(0)
        p_l_edge_index = torch.cat([p_l_edge_target, p_l_edge_source], dim=0)
        edge_index = torch.cat([p_edge_index, l_edge_index, p_l_edge_index], dim=1)
        row, col = edge_index[1], edge_index[0]

        return torch.stack([p_edge_index[1, :], p_edge_index[0, :]], dim=0), torch.stack(
            [l_edge_index[1, :], l_edge_index[0, :]], dim=0), torch.stack([p_l_edge_index[1, :], p_l_edge_index[0, :]],
                                                                          dim=0), torch.stack([row, col], dim=0)

    def _connect_edge_pre(self, pos_protein, pos_ligand, batch_protein, batch_ligand):

        p_atom_num = pos_protein.size(0)

        l_edge_index = knn(pos_ligand, pos_ligand, self.only_l + 1, batch_ligand, batch_ligand)

        l_inner_mask = l_edge_index[0] != l_edge_index[1]

        l_edge_index = l_edge_index[:, l_inner_mask] + p_atom_num

        row, col = l_edge_index[1], l_edge_index[0]

        return torch.stack([row, col], dim=0)

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1  #
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type


    def forward(self, h, x, mask_ligand, batch, pos_protein, pos_ligand, batch_protein, batch_ligand, return_all=False,
                fix_x=False, pre_train=False):

        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            p_edge_index, l_edge_index, p_l_edge_index, edge_index = self._connect_edge_sep(pos_protein, pos_ligand,
                                                                                            batch_protein, batch_ligand)
            src, dst = edge_index

            # edge type (dim: 4)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)  #
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            p_edge_end=p_edge_index.shape[1]
            l_edge_end=l_edge_index.shape[1]+p_edge_end


            for l_idx, _ in enumerate(self.base_block_ligand):
                h_protein = self.base_block_protein[l_idx](h, x, edge_type[:p_edge_end], p_edge_index, mask_ligand, e_w=e_w[:p_edge_end,:], fix_x=fix_x,protein_mode=True)
                h_protein=h_protein [~mask_ligand]
                h_ligand,delta_x = self.base_block_ligand[l_idx](h, x, edge_type[p_edge_end:l_edge_end], l_edge_index, mask_ligand, e_w=e_w[p_edge_end:l_edge_end,:], fix_x=fix_x,protein_mode=False)
                h_ligand=h_ligand [mask_ligand]
                delta_x = delta_x [mask_ligand]

                ##method1:
                delta_x_mean=scatter(delta_x,batch_ligand,dim=0,reduce='mean')
                delta_x=delta_x-delta_x_mean[batch_ligand]
                #
                ##method2:
                ##pass

                ##Message Passing
                x[mask_ligand]=x[mask_ligand]+delta_x
                h=torch.cat([h_protein,h_ligand],dim=0)

                ##Interaction
                h,delta_x_pl=self.base_block_pl[l_idx](h, x, edge_type[l_edge_end:], p_l_edge_index, mask_ligand, e_w=e_w[l_edge_end:,:], fix_x=fix_x,protein_mode=False)
                delta_x_pl=delta_x_pl [mask_ligand]

                ##method1:
                # deta_x_pl_mean=scatter(delta_x_pl,batch_ligand,dim=0,reduce='mean')
                # x[mask_ligand]=x[mask_ligand]+deta_x_pl_mean[batch_ligand]

                ##method2:
                x[mask_ligand]=x[mask_ligand]+delta_x_pl



            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}  #
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs


    def InLayer_edge_connect(self, pos_protein, pos_ligand, batch_protein, batch_ligand):
        p_atom_num = pos_protein.size(0)


        l_edge_index = knn(pos_ligand, pos_ligand, self.l_k + 1, batch_ligand, batch_ligand)
        l_inner_mask = l_edge_index[0] != l_edge_index[1]
        l_edge_index = l_edge_index[:, l_inner_mask] + p_atom_num

        p_l_edge_index = knn(pos_protein, pos_ligand, self.pl_k, batch_protein, batch_ligand)
        p_l_edge_target = torch.cat([p_l_edge_index[0] + p_atom_num, p_l_edge_index[1]]).unsqueeze(0)
        p_l_edge_source = torch.cat([p_l_edge_index[1], p_l_edge_index[0] + p_atom_num]).unsqueeze(0)
        p_l_edge_index = torch.cat([p_l_edge_target, p_l_edge_source], dim=0)
        edge_index = torch.cat([l_edge_index, p_l_edge_index], dim=1)
        row, col = edge_index[1], edge_index[0]

        return torch.stack(
            [l_edge_index[1, :], l_edge_index[0, :]], dim=0), torch.stack(
            [p_l_edge_index[1, :], p_l_edge_index[0, :]],
            dim=0), torch.stack([row, col], dim=0)



class UniTransformerO2TwoUpdateGeneral_Interact_Dynamic(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='knn', ew_net_type='global',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=False, sync_twoup=False,
                 connect_type='SE'):  ##SA,SE
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks  ##1
        self.num_layers = num_layers  ##9
        self.hidden_dim = hidden_dim  ##128
        self.n_heads = n_heads  ##1
        self.num_r_gaussian = num_r_gaussian  ##50
        self.edge_feat_dim = edge_feat_dim  ##0
        self.act_fn = act_fn  ##relu
        self.norm = norm  ##True
        self.num_node_types = num_node_types  ##8
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none] ##knn
        self.k = k
        self.connect_type = connect_type
        ###add
        self.p_k = 24
        self.l_k = 32#48
        self.pl_k = 32#48 32
        self.only_l = 40
        ###end add
        self.ew_net_type = ew_net_type  # [r, m, none] global

        self.num_x2h = num_x2h  ##1
        self.num_h2x = num_h2x  ##1
        self.num_init_x2h = num_init_x2h  ##1
        self.num_init_h2x = num_init_h2x  ##0
        self.r_max = r_max  ##10
        self.x2h_out_fc = x2h_out_fc  ##True
        self.sync_twoup = sync_twoup  ##False
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)  ##
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim,act_fn=self.act_fn)  ##20-128

        self.init_h_emb_layer = self._build_init_h_layer()  ##
        self.base_block_ligand = self._build_share_blocks()
        self.base_block_protein = self._build_share_blocks()
        self.base_block_pl = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block_ligand.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral_Inter(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):  #
            layer = AttentionLayerO2TwoUpdateNodeGeneral_Inter(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    def _connect_edge_sep(self, pos_protein, pos_ligand, batch_protein, batch_ligand):

        p_atom_num = pos_protein.size(0)

        p_edge_index = knn(pos_protein, pos_protein, self.p_k + 1, batch_protein,
                           batch_protein)  #
        l_edge_index = knn(pos_ligand, pos_ligand, self.l_k + 1, batch_ligand, batch_ligand)
        p_inner_mask = p_edge_index[0] != p_edge_index[1]
        l_inner_mask = l_edge_index[0] != l_edge_index[1]
        p_edge_index = p_edge_index[:, p_inner_mask]
        l_edge_index = l_edge_index[:, l_inner_mask] + p_atom_num

        p_l_edge_index = knn(pos_protein, pos_ligand, self.pl_k, batch_protein, batch_ligand)
        p_l_edge_target = torch.cat([p_l_edge_index[0] + p_atom_num, p_l_edge_index[1]]).unsqueeze(0)
        p_l_edge_source = torch.cat([p_l_edge_index[1], p_l_edge_index[0] + p_atom_num]).unsqueeze(0)
        p_l_edge_index = torch.cat([p_l_edge_target, p_l_edge_source], dim=0)
        edge_index = torch.cat([p_edge_index, l_edge_index, p_l_edge_index], dim=1)
        row, col = edge_index[1], edge_index[0]

        return torch.stack([p_edge_index[1, :], p_edge_index[0, :]], dim=0), torch.stack(
            [l_edge_index[1, :], l_edge_index[0, :]], dim=0), torch.stack([p_l_edge_index[1, :], p_l_edge_index[0, :]],
                                                                          dim=0), torch.stack([row, col], dim=0)

    def _connect_edge_pre(self, pos_protein, pos_ligand, batch_protein, batch_ligand):

        p_atom_num = pos_protein.size(0)


        l_edge_index = knn(pos_ligand, pos_ligand, self.only_l + 1, batch_ligand, batch_ligand)

        l_inner_mask = l_edge_index[0] != l_edge_index[1]

        l_edge_index = l_edge_index[:, l_inner_mask] + p_atom_num

        row, col = l_edge_index[1], l_edge_index[0]

        return torch.stack([row, col], dim=0)

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1  #
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type


    def forward(self, h, x, mask_ligand, batch, pos_protein, pos_ligand, batch_protein, batch_ligand, return_all=False,
                fix_x=False, pre_train=False):

        all_x = [x]
        all_h = [h]

        for b_idx in range(self.num_blocks):
            p_edge_index, l_edge_index, p_l_edge_index, edge_index = self._connect_edge_sep(pos_protein, pos_ligand,
                                                                                            batch_protein, batch_ligand)
            src, dst = edge_index

            # edge type (dim: 4)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)  #
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            p_edge_end=p_edge_index.shape[1]
            l_edge_end=l_edge_index.shape[1]+p_edge_end


            for l_idx, _ in enumerate(self.base_block_ligand):
                h_protein = self.base_block_protein[l_idx](h, x, edge_type[:p_edge_end], p_edge_index, mask_ligand, e_w=e_w[:p_edge_end,:], fix_x=fix_x,protein_mode=True)
                h_protein=h_protein [~mask_ligand]
                h_ligand,delta_x = self.base_block_ligand[l_idx](h, x, edge_type[p_edge_end:l_edge_end], l_edge_index, mask_ligand, e_w=e_w[p_edge_end:l_edge_end,:], fix_x=fix_x,protein_mode=False)
                h_ligand=h_ligand [mask_ligand]
                delta_x = delta_x [mask_ligand]

                ##method1:
                delta_x_mean=scatter(delta_x,batch_ligand,dim=0,reduce='mean')
                delta_x=delta_x-delta_x_mean[batch_ligand]
                #
                ##method2:
                ##pass

                ##Message Passing
                x[mask_ligand]=x[mask_ligand]+delta_x
                h=torch.cat([h_protein,h_ligand],dim=0)

                ##Interaction
                h,delta_x_pl=self.base_block_pl[l_idx](h, x, edge_type[l_edge_end:], p_l_edge_index, mask_ligand, e_w=e_w[l_edge_end:,:], fix_x=fix_x,protein_mode=False)
                delta_x_pl=delta_x_pl [mask_ligand]

                ##method1:
                # deta_x_pl_mean=scatter(delta_x_pl,batch_ligand,dim=0,reduce='mean')
                # x[mask_ligand]=x[mask_ligand]+deta_x_pl_mean[batch_ligand]

                ##method2:
                x[mask_ligand]=x[mask_ligand]+delta_x_pl



            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}  #
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs


    def InLayer_edge_connect(self, pos_protein, pos_ligand, batch_protein, batch_ligand):
        p_atom_num = pos_protein.size(0)


        l_edge_index = knn(pos_ligand, pos_ligand, self.l_k + 1, batch_ligand, batch_ligand)
        l_inner_mask = l_edge_index[0] != l_edge_index[1]
        l_edge_index = l_edge_index[:, l_inner_mask] + p_atom_num

        p_l_edge_index = knn(pos_protein, pos_ligand, self.pl_k, batch_protein, batch_ligand)
        p_l_edge_target = torch.cat([p_l_edge_index[0] + p_atom_num, p_l_edge_index[1]]).unsqueeze(0)
        p_l_edge_source = torch.cat([p_l_edge_index[1], p_l_edge_index[0] + p_atom_num]).unsqueeze(0)
        p_l_edge_index = torch.cat([p_l_edge_target, p_l_edge_source], dim=0)
        edge_index = torch.cat([l_edge_index, p_l_edge_index], dim=1)
        row, col = edge_index[1], edge_index[0]

        return torch.stack(
            [l_edge_index[1, :], l_edge_index[0, :]], dim=0), torch.stack(
            [p_l_edge_index[1, :], p_l_edge_index[0, :]],
            dim=0), torch.stack([row, col], dim=0)
