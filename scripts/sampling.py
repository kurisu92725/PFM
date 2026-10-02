import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import scatter
from tqdm.auto import tqdm

from datasets.pl_data import FOLLOW_BATCH
from utils.evaluation import atom_num


def _prediction(output):
    if not isinstance(output, tuple) or not output:
        raise RuntimeError('Unexpected model output.')
    return output[0]


def _prepare_sampling(device, one_data, prior, num_mole):
    batch = Batch.from_data_list(
        [one_data.clone() for _ in range(num_mole)],
        follow_batch=FOLLOW_BATCH,
    ).to(device)
    protein_pos = batch.protein_pos
    protein_v = batch.protein_atom_feature.float()
    b_p = batch.protein_element_batch
    p_mean = scatter(protein_pos, b_p, dim=0, reduce='mean')
    protein_pos_input = protein_pos - p_mean[b_p]

    pocket_size = atom_num.get_space_size(
        one_data.protein_pos.detach().cpu().numpy()
    )
    ligand_num_atoms = [
        int(atom_num.sample_atom_num(pocket_size)) for _ in range(num_mole)
    ]
    b_l = torch.repeat_interleave(
        torch.arange(num_mole, device=device),
        torch.as_tensor(ligand_num_atoms, device=device),
    )
    p_xh = torch.cat([protein_pos_input, protein_v], dim=-1)
    z_x0, z_h0 = prior.sample(ligand_num_atoms, device)
    return ligand_num_atoms, b_l, b_p, p_mean, p_xh, z_x0, z_h0


def _wrap_predictors(x_pre, h_pre, b_l, p_xh, b_p):
    x_pre.forward = x_pre.wrap_forward(
        b_l,
        p_xh,
        b_p,
        node_mask=None,
        edge_mask=None,
        context=None,
        reverse=False,
        pre_train=False,
    )
    h_pre.forward = h_pre.wrap_forward(
        b_l,
        p_xh,
        b_p,
        node_mask=None,
        edge_mask=None,
        context=None,
        reverse=False,
        pre_train=False,
    )


def _initial_atom_state(z_h0, const_k):
    z_h0 = z_h0 * const_k
    z_h0_v = F.one_hot(
        torch.argmax(F.softmax(z_h0, dim=-1), dim=1),
        num_classes=12,
    ).float()
    return z_h0, z_h0_v


def _perturbations(
    t,
    z_x0,
    z_h0_v,
    pred_z1,
    pred_h_idx,
    x_pre,
    h_pre,
    x_pare,
    h_para,
    const_k,
):
    pred_x_bar = _prediction(
        x_pre(
            t,
            torch.cat(
                [z_x0, F.one_hot(pred_h_idx, num_classes=12).float()],
                dim=-1,
            ),
        )
    )[:, :3] * x_pare
    pred_h_bar = _prediction(
        h_pre(t, torch.cat([pred_z1[:, :3], z_h0_v], dim=-1))
    )[:, 3:]
    pred_h_bar = torch.argmax(F.softmax(pred_h_bar, dim=-1), dim=1)
    pred_h_bar = (
        F.one_hot(pred_h_bar, num_classes=12).float() * 2 * const_k - const_k
    ) * h_para
    return pred_x_bar, pred_h_bar


def _unbatch(pred_pos, pred_v, ligand_num_atoms):
    ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
    pos_np = pred_pos.detach().cpu().numpy().astype(np.float64)
    v_np = pred_v.detach().cpu().numpy()
    ligand_p = [
        pos_np[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]]
        for k in range(len(ligand_num_atoms))
    ]
    ligand_v = [
        v_np[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]]
        for k in range(len(ligand_num_atoms))
    ]
    return ligand_p, ligand_v


def _pfm_position_velocity(x_t, pred_x1, pred_x_bar, t):
    denominator = torch.clamp(1 - t, min=1.0e-5, max=1.0)
    return (pred_x1 - x_t) / denominator + (1 - t) * pred_x_bar


def _pfm_g_position_velocity(
    x_t,
    pred_x1_cond,
    pred_x1_uncond,
    pred_x_bar_cond,
    pred_x_bar_uncond,
    t,
    guidance_scale,
):
    denominator = torch.clamp(1 - t, min=1.0e-5, max=1.0)
    v_cond = (pred_x1_cond - x_t) / denominator
    v_uncond = (pred_x1_uncond - x_t) / denominator
    return (
        v_uncond
        + guidance_scale * (v_cond - v_uncond)
        + (guidance_scale - 1) * t
        * (pred_x_bar_cond - pred_x_bar_uncond)
    )


def sample_ode_pfm(
    device,
    one_data,
    x_pre,
    h_pre,
    x_pare,
    h_para,
    prior,
    net_dynamics,
    num_mole=10,
    num_steps=100,
):
    (
        ligand_num_atoms,
        b_l,
        b_p,
        p_mean,
        p_xh,
        z_x0,
        z_h0,
    ) = _prepare_sampling(device, one_data, prior, num_mole)

    net_dynamics.forward = net_dynamics.wrap_forward(
        b_l,
        p_xh,
        b_p,
        node_mask=None,
        edge_mask=None,
        context=None,
        reverse=False,
        pre_train=False,
    )
    _wrap_predictors(x_pre, h_pre, b_l, p_xh, b_p)

    const_k = 5.0
    z_h0, z_h0_v = _initial_atom_state(z_h0, const_k)
    z_x1 = z_x0.clone()
    z_h1 = z_h0.clone()
    z_h1v = z_h0_v.clone()

    num_steps = int(num_steps)
    if num_steps < 2:
        raise ValueError(f'num_steps must be at least 2, got {num_steps}')
    ts = torch.linspace(1.0e-2, 1.0, num_steps, device=device)
    t_1 = ts[0]

    for t_2 in tqdm(ts[1:], leave=False):
        pre_z1 = _prediction(
            net_dynamics(t_1, torch.cat([z_x1, z_h1v], dim=-1))
        )
        z_htv = torch.argmax(F.softmax(pre_z1[:, 3:], dim=-1), dim=1)
        h_t = F.one_hot(z_htv, num_classes=12).float() * 2 * const_k - const_k
        pre_x1_bar, pre_h1_bar = _perturbations(
            t_1,
            z_x0,
            z_h0_v,
            pre_z1,
            z_htv,
            x_pre,
            h_pre,
            x_pare,
            h_para,
            const_k,
        )

        dt = (t_2 - t_1).to(z_x1).reshape(1, 1)
        v_x1 = _pfm_position_velocity(
            z_x1,
            pre_z1[:, :3],
            pre_x1_bar,
            t_1,
        )
        z_x1 = z_x1 + v_x1 * dt
        z_h1 = z_h1 + (
            h_t - z_h0 + (1 - 2 * t_1) * pre_h1_bar
        ) * dt
        z_h1v = F.one_hot(
            torch.argmax(F.softmax(z_h1, dim=-1), dim=1),
            num_classes=12,
        ).float()
        t_1 = t_2

    pre_final_z = _prediction(
        net_dynamics(ts[-1], torch.cat([z_x1, z_h1v], dim=-1))
    )
    pre_pos = pre_final_z[:, :3] + p_mean[b_l]
    pre_v = torch.argmax(F.softmax(pre_final_z[:, 3:], dim=-1), dim=1) + 1
    return _unbatch(pre_pos, pre_v, ligand_num_atoms)


def sample_ode_pfm_g(
    device,
    one_data,
    x_pre,
    h_pre,
    x_pare,
    h_para,
    prior,
    net_dynamics,
    num_mole=10,
    num_steps=100,
    guidance_scale=5.5,
    guidance_target=1.0,
):
    (
        ligand_num_atoms,
        b_l,
        b_p,
        p_mean,
        p_xh,
        z_x0,
        z_h0,
    ) = _prepare_sampling(device, one_data, prior, num_mole)
    _wrap_predictors(x_pre, h_pre, b_l, p_xh, b_p)

    const_k = 5.0
    z_h0, z_h0_v = _initial_atom_state(z_h0, const_k)
    z_x1 = z_x0.clone()
    z_h1 = z_h0.clone()
    z_h1v = z_h0_v.clone()

    num_steps = int(num_steps)
    if num_steps < 2:
        raise ValueError(f'num_steps must be at least 2, got {num_steps}')
    guidance_scale = float(guidance_scale)
    context = torch.full(
        (num_mole, 1),
        float(guidance_target),
        device=device,
    )

    ts = torch.linspace(1.0e-2, 1.0, num_steps, device=device)
    t_1 = ts[0]

    for t_2 in tqdm(ts[1:], leave=False):
        net_dynamics.forward = net_dynamics.wrap_forward(
            b_l,
            p_xh,
            b_p,
            node_mask=1,
            edge_mask=None,
            context=context,
            reverse=False,
            pre_train=False,
        )
        pre_z1_g = _prediction(
            net_dynamics(t_1, torch.cat([z_x1, z_h1v], dim=-1))
        )
        z_htv_g = torch.argmax(F.softmax(pre_z1_g[:, 3:], dim=-1), dim=1)
        h_t_g = (
            F.one_hot(z_htv_g, num_classes=12).float() * 2 * const_k - const_k
        )

        net_dynamics.forward = net_dynamics.wrap_forward(
            b_l,
            p_xh,
            b_p,
            node_mask=0,
            edge_mask=None,
            context=context,
            reverse=False,
            pre_train=False,
        )
        pre_z1_ng = _prediction(
            net_dynamics(t_1, torch.cat([z_x1, z_h1v], dim=-1))
        )
        z_htv_ng = torch.argmax(F.softmax(pre_z1_ng[:, 3:], dim=-1), dim=1)

        pre_x1_bar_ng, _ = _perturbations(
            t_1,
            z_x0,
            z_h0_v,
            pre_z1_ng,
            z_htv_ng,
            x_pre,
            h_pre,
            x_pare,
            h_para,
            const_k,
        )
        pre_x1_bar_g, pre_h1_bar = _perturbations(
            t_1,
            z_x0,
            z_h0_v,
            pre_z1_g,
            z_htv_g,
            x_pre,
            h_pre,
            x_pare,
            h_para,
            const_k,
        )

        v_x1 = _pfm_g_position_velocity(
            z_x1,
            pre_z1_g[:, :3],
            pre_z1_ng[:, :3],
            pre_x1_bar_g,
            pre_x1_bar_ng,
            t_1,
            guidance_scale,
        )

        dt = (t_2 - t_1).to(z_x1).reshape(1, 1)
        z_x1 = z_x1 + v_x1 * dt
        z_h1 = z_h1 + (
            h_t_g - z_h0 + (1 - 2 * t_1) * pre_h1_bar
        ) * dt
        z_h1v = F.one_hot(
            torch.argmax(F.softmax(z_h1, dim=-1), dim=1),
            num_classes=12,
        ).float()
        t_1 = t_2

    net_dynamics.forward = net_dynamics.wrap_forward(
        b_l,
        p_xh,
        b_p,
        node_mask=1,
        edge_mask=None,
        context=context,
        reverse=False,
        pre_train=False,
    )
    pre_final_z = _prediction(
        net_dynamics(ts[-1], torch.cat([z_x1, z_h1v], dim=-1))
    )
    pre_pos = pre_final_z[:, :3] + p_mean[b_l]
    pre_v = torch.argmax(F.softmax(pre_final_z[:, 3:], dim=-1), dim=1) + 1
    return _unbatch(pre_pos, pre_v, ligand_num_atoms)
