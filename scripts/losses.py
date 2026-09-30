import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

from utils.remove_mean import remove_mean_crossdock


def type2prob_traj(norm_k, x):
    return F.one_hot(x, 12).float() * 2 * norm_k - norm_k


def sample_v_from_softmax(p):
    return torch.multinomial(p + 1e-8, 1).squeeze(1)


def sym_t_sampler(batch_ligand, pos):
    count = batch_ligand.max().item() + 1
    if count % 4:
        return torch.rand(count).type_as(pos)
    base = torch.rand(count // 4).type_as(pos) / 4
    times = torch.cat([base + i / 4 for i in range(4)])
    return times[torch.randperm(count, device=pos.device)]


def _batch_inputs(config, batch):
    protein_pos = batch.protein_pos + torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
    protein_v = batch.protein_atom_feature.float()
    batch_protein = batch.protein_element_batch
    atom_type_full = batch.ligand_atom_feature_full
    supported = atom_type_full != 0
    if not supported.any():
        raise ValueError('Ligand contains no supported atoms.')
    ligand_pos = batch.ligand_pos[supported]
    ligand_v = atom_type_full[supported] - 1
    batch_ligand = batch.ligand_element_batch[supported]
    if (ligand_v < 0).any() or (ligand_v >= 12).any():
        raise ValueError('Ligand atom type is outside the 12 supported classes.')
    old2new = torch.full((len(supported),), -1, dtype=torch.long, device=ligand_pos.device)
    old2new[supported] = torch.arange(len(ligand_v), device=ligand_pos.device)
    source, target = batch.ligand_bond_index
    keep_bond = supported[source] & supported[target]
    ligand_bond_index = torch.stack([
        old2new[source[keep_bond]], old2new[target[keep_bond]]
    ])
    ligand_bond = batch.ligand_bond_type[keep_bond] - 1
    ligand_bond_batch = batch.ligand_bond_type_batch[keep_bond]
    protein_pos, ligand_pos = remove_mean_crossdock(
        protein_pos, ligand_pos, batch_protein, batch_ligand
    )
    p_xh = torch.cat([protein_pos, protein_v], dim=-1)
    return (protein_pos, ligand_pos, ligand_v, batch_protein, batch_ligand,
            ligand_bond_index, ligand_bond, ligand_bond_batch, p_xh)


def _surface_loss(protein_pos, batch_protein, pred_pos, batch_ligand, sigma, gamma):
    losses = []
    for i in range(batch_ligand.max().item() + 1):
        pocket = protein_pos[batch_protein == i]
        ligand = pred_pos[batch_ligand == i]
        distance = (pocket[None] - ligand[:, None]).square().sum(dim=-1)
        surface = -sigma * torch.log(1e-3 + torch.exp(-distance / sigma).sum(dim=1))
        losses.append(torch.clamp(gamma - surface, min=0).mean())
    return torch.stack(losses).unsqueeze(1)


def _field_loss(config, inputs, prediction, bond_prediction=None, score_prediction=None, score_target=None):
    protein_pos, ligand_pos, ligand_v, batch_protein, batch_ligand, bond_index, bond_type, bond_batch, _ = inputs
    hp = config.train.loss_hyper
    pos_loss = scatter(
        (prediction[:, :3] - ligand_pos).square().sum(dim=1, keepdim=True),
        batch_ligand, dim=0, reduce='mean',
    )
    if bond_index.size(1):
        actual = (ligand_pos[bond_index[0]] - ligand_pos[bond_index[1]]).square().sum(dim=-1)
        predicted = (prediction[bond_index[0], :3] - prediction[bond_index[1], :3]).square().sum(dim=-1)
        distance_loss = scatter(
            (predicted - actual).abs().unsqueeze(1), bond_batch, dim=0,
            dim_size=pos_loss.size(0), reduce='mean',
        )
        pos_loss = pos_loss + hp.lambda_edgeL * distance_loss
    if hp.lambda_surf:
        pos_loss = pos_loss + hp.lambda_surf * _surface_loss(
            protein_pos, batch_protein, prediction[:, :3], batch_ligand,
            hp.surf_sigma, hp.surf_gamma,
        )
    h_loss = F.cross_entropy(prediction[:, 3:], ligand_v)
    total = pos_loss.mean() + hp.lambda_h * h_loss
    if bond_prediction is not None and hp.lambda_bond_tp:
        total = total + hp.lambda_bond_tp * F.cross_entropy(bond_prediction, bond_type)
    lambda_vina = float(getattr(hp, 'lambda_vina', 0.0))
    if score_prediction is not None and lambda_vina:
        total = total + hp.lambda_vina * F.mse_loss(
            score_prediction.reshape(-1), score_target.reshape(-1)
        )
    return total, pos_loss.mean(), h_loss


def compute_loss_pfm(config, x_bar_predictor, h_bar_predictor, FM, FM_align, prior, net_dynamics, batch, test):
    inputs = _batch_inputs(config, batch)
    _, ligand_pos, ligand_v, batch_protein, batch_ligand, bond_index, _, _, p_xh = inputs
    norm_k = config.train.loss_hyper.norm_k
    net_dynamics.forward = net_dynamics.wrap_forward(
        batch_ligand, p_xh, batch_protein, None, bond_index, None, False, False
    )
    pos_0, lh_0 = prior.sample(ligand_pos.size(0), ligand_pos.device)
    lh_0 = lh_0 * norm_k
    t = sym_t_sampler(batch_ligand, pos_0)
    target = torch.cat([ligand_pos, type2prob_traj(norm_k, ligand_v)], dim=-1)
    x0_align = FM_align.sample_location_and_conditional_flow(
        torch.cat([pos_0, lh_0], dim=-1), target, batch_ligand, t=t
    )
    x_bar_input = torch.cat([x0_align[:, :3], F.one_hot(ligand_v, 12).float()], dim=-1)
    h_bar_input = torch.cat([
        ligand_pos,
        F.one_hot(sample_v_from_softmax(F.softmax(lh_0, dim=-1)), 12).float(),
    ], dim=-1)
    with torch.no_grad():
        x_bar_predictor.forward = x_bar_predictor.wrap_forward(
            batch_ligand, p_xh, batch_protein, None, None, None, False, False
        )
        h_bar_predictor.forward = h_bar_predictor.wrap_forward(
            batch_ligand, p_xh, batch_protein, None, None, None, False, False
        )
        x_bar = x_bar_predictor(t, x_bar_input)[0][:, :3]
        h_bar = h_bar_predictor(t, h_bar_input)[0][:, 3:]
        h_bar = sample_v_from_softmax(F.softmax(h_bar, dim=-1))
        total_bar = torch.cat([x_bar, type2prob_traj(norm_k, h_bar)], dim=-1)
    _, state, _ = FM(x0_align, target, total_bar, batch_ligand, t=t)
    zt = torch.cat([
        state[:, :3],
        F.one_hot(sample_v_from_softmax(F.softmax(state[:, 3:], dim=-1)), 12).float(),
    ], dim=-1)
    output = net_dynamics(t, zt)
    total, pos_loss, h_loss = _field_loss(config, inputs, output[0], output[1])
    if test:
        net_dynamics.forward = net_dynamics.unwrap_forward()
        return total, total, pos_loss, h_loss, F.softmax(output[0][:, 3:], dim=-1), ligand_v
    return total, total, pos_loss, h_loss


def compute_loss_pfm_g(config, x_bar_predictor, h_bar_predictor, FM, FM_align, prior, net_dynamics, batch, test):
    inputs = _batch_inputs(config, batch)
    _, ligand_pos, ligand_v, batch_protein, batch_ligand, bond_index, _, _, p_xh = inputs
    norm_k = config.train.loss_hyper.norm_k
    score = batch.vina_dock.reshape(-1).to(ligand_pos)
    score = ((0.0 - score.clamp(max=0.0)) / 17.203).clamp(0.0, 1.0).unsqueeze(1)
    net_dynamics.forward = net_dynamics.wrap_forward(
        batch_ligand, p_xh, batch_protein, 2, bond_index, score, False, False
    )
    pos_0, lh_0 = prior.sample(ligand_pos.size(0), ligand_pos.device)
    lh_0 = lh_0 * norm_k
    t = sym_t_sampler(batch_ligand, pos_0)
    target = torch.cat([ligand_pos, type2prob_traj(norm_k, ligand_v)], dim=-1)
    x0_align = FM_align.sample_location_and_conditional_flow(
        torch.cat([pos_0, lh_0], dim=-1), target, batch_ligand, t=t
    )
    _, state, _ = FM(x0_align, target, torch.zeros_like(target), batch_ligand, t=t)
    zt = torch.cat([
        state[:, :3],
        F.one_hot(sample_v_from_softmax(F.softmax(state[:, 3:], dim=-1)), 12).float(),
    ], dim=-1)
    output = net_dynamics(t, zt)
    total, pos_loss, h_loss = _field_loss(
        config, inputs, output[0], output[1], output[2], score
    )
    if test:
        net_dynamics.forward = net_dynamics.unwrap_forward()
        return total, total, pos_loss, h_loss, F.softmax(output[0][:, 3:], dim=-1), ligand_v
    return total, total, pos_loss, h_loss


def _predictor_loss(config, prior, net_dynamics, batch, test, mode):
    inputs = _batch_inputs(config, batch)
    _, ligand_pos, ligand_v, batch_protein, batch_ligand, bond_index, _, _, p_xh = inputs
    norm_k = config.train.loss_hyper.norm_k
    net_dynamics.forward = net_dynamics.wrap_forward(
        batch_ligand, p_xh, batch_protein, None, bond_index, None, False, False
    )
    pos_0, lh_0 = prior.sample(ligand_pos.size(0), ligand_pos.device)
    lh_0 = lh_0 * norm_k
    t = sym_t_sampler(batch_ligand, pos_0)
    if mode == 'xp':
        noisy = type2prob_traj(norm_k, ligand_v) + torch.randn_like(lh_0) * 2.5
        atom_idx = sample_v_from_softmax(F.softmax(noisy, dim=-1))
        zt = torch.cat([pos_0, F.one_hot(atom_idx, 12).float()], dim=-1)
    else:
        atom_idx = sample_v_from_softmax(F.softmax(lh_0, dim=-1))
        zt = torch.cat([
            ligand_pos + torch.randn_like(ligand_pos) * 0.5,
            F.one_hot(atom_idx, 12).float(),
        ], dim=-1)
    prediction = net_dynamics(t, zt)[0]
    total, pos_loss, h_loss = _field_loss(config, inputs, prediction)
    if test:
        net_dynamics.forward = net_dynamics.unwrap_forward()
        return total, total, pos_loss, h_loss, F.softmax(prediction[:, 3:], dim=-1), ligand_v
    return total, total, pos_loss, h_loss


def compute_loss_x_predictor(config, x_bar_predictor, h_bar_predictor, FM, FM_align, prior, net_dynamics, batch, test):
    return _predictor_loss(config, prior, net_dynamics, batch, test, 'xp')


def compute_loss_h_predictor(config, x_bar_predictor, h_bar_predictor, FM, FM_align, prior, net_dynamics, batch, test):
    return _predictor_loss(config, prior, net_dynamics, batch, test, 'hp')
