import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import losses
import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from egnn.DotGat import DotPre_FM
from scripts.flow_matching_utils import CFM_CD_PERTURB, POS_HA
from scripts.prior_distributions import PositionFeaturePrior_PFM
from utils.checkpoint import configure_model_for_variant, load_predictor_state


def resolve_path(path, config_path):
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(config_path), path))


def get_auroc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weighted = 0.0
    count = 0
    for atom_type in np.unique(y_true):
        labels = y_true == atom_type
        if labels.all() or not labels.any():
            continue
        size = int(labels.sum())
        weighted += roc_auc_score(labels, y_pred[:, atom_type]) * size
        count += size
    return weighted / count if count else float('nan')


def save_checkpoint(path, epoch, iter_num, net_dynamics, FM, optimizer, scheduler):
    torch.save({
        'epoch': epoch,
        'iter_num': iter_num,
        'net_dynamics_state_dict': net_dynamics.state_dict(),
        'FM_state_dict': FM.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_iter_test', type=int, default=400)
    parser.add_argument('--interrupt_protection', type=int, default=2000)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = misc.load_config(config_path)
    config.data.path = resolve_path(config.data.path, config_path)
    config.data.split = resolve_path(config.data.split, config_path)
    if getattr(config.data, 'index_path', None) is not None:
        config.data.index_path = resolve_path(config.data.index_path, config_path)
    for predictor_key in ('x_predictor', 'h_predictor'):
        predictor_path = getattr(config.data, predictor_key, None)
        if predictor_path is not None:
            config.data[predictor_key] = resolve_path(predictor_path, config_path)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    misc.seed_all(int(config.train.seed))
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    logger.info(args)
    logger.info(config)

    transform_list = [
        trans.FeaturizeProteinAtom(),
        trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode),
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    dataset_kwargs = {'transform': Compose(transform_list)}
    if getattr(config.data, 'index_path', None) is not None:
        dataset_kwargs['index_path'] = config.data.index_path
    dataset, subsets = get_dataset(config=config.data, **dataset_kwargs)
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    loader_kwargs = {
        'batch_size': int(config.train.batch_size),
        'num_workers': int(config.train.num_workers),
        'follow_batch': FOLLOW_BATCH,
        'exclude_keys': ['ligand_nbh_list'],
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    training_mode = str(config.train.training_mode).lower()
    if training_mode not in {'xp', 'hp', 'pfm', 'pfm_g'}:
        raise ValueError('training_mode must be xp, hp, pfm, or pfm_g')
    variant = 'pfm_g' if training_mode == 'pfm_g' else 'pfm'
    configure_model_for_variant(config.model, variant)
    x_bar_predictor = DotPre_FM(config.modelp, 27, 12)
    h_bar_predictor = DotPre_FM(config.modelp, 27, 12)
    model_config = config.modelp if training_mode in {'xp', 'hp'} else config.model
    net_dynamics = DotPre_FM(model_config, 27, 12).to(args.device)

    if training_mode in {'pfm', 'pfm_g'}:
        load_predictor_state(x_bar_predictor, config.data.x_predictor, args.device)
        load_predictor_state(h_bar_predictor, config.data.h_predictor, args.device)
        x_bar_predictor = x_bar_predictor.to(args.device).eval()
        h_bar_predictor = h_bar_predictor.to(args.device).eval()
        for predictor in (x_bar_predictor, h_bar_predictor):
            for parameter in predictor.parameters():
                parameter.requires_grad = False

    loss_functions = {
        'xp': losses.compute_loss_x_predictor,
        'hp': losses.compute_loss_h_predictor,
        'pfm': losses.compute_loss_pfm,
        'pfm_g': losses.compute_loss_pfm_g,
    }
    get_losses = loss_functions[training_mode]
    FM = CFM_CD_PERTURB(config.train.sigma_pos).to(args.device)
    FM_align = POS_HA(config.train.sigma_pos)
    prior = PositionFeaturePrior_PFM(n_dim=3, in_node_nf=12)
    optimizer = utils_train.get_optimizer(config.train.optimizer, net_dynamics)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=args.device)
        net_dynamics.load_state_dict(checkpoint['net_dynamics_state_dict'], strict=True)
        FM.load_state_dict(checkpoint['FM_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1

    def validate():
        net_dynamics.eval()
        totals = []
        positions = []
        atom_types = []
        all_pred_v = []
        all_true_v = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                output = get_losses(
                    config, x_bar_predictor, h_bar_predictor, FM, FM_align,
                    prior, net_dynamics, batch, test=True,
                )
                total_loss, _, pos_loss, h_loss, pre_v, real_v = output
                totals.append(float(total_loss))
                positions.append(float(pos_loss))
                atom_types.append(float(h_loss))
                all_pred_v.append(pre_v.detach().cpu().numpy())
                all_true_v.append(real_v.detach().cpu().numpy())
        auroc = get_auroc(
            np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0)
        )
        return np.mean(totals), np.mean(positions), np.mean(atom_types), auroc

    best_total_val = float('inf')
    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        epoch_start = time.time()
        for iter_num, batch in enumerate(tqdm(train_loader)):
            net_dynamics.train()
            optimizer.zero_grad()
            batch = batch.to(args.device)
            total_loss, field_loss, pos_loss, h_loss = get_losses(
                config, x_bar_predictor, h_bar_predictor, FM, FM_align,
                prior, net_dynamics, batch, test=False,
            )
            total_loss.backward()
            grad_norm = clip_grad_norm_(
                net_dynamics.parameters(), config.train.max_grad_norm
            )
            optimizer.step()

            if iter_num % args.interrupt_protection == 0:
                save_checkpoint(
                    os.path.join(checkpoint_dir, 'protection.pth'),
                    epoch, iter_num, net_dynamics, FM, optimizer, scheduler,
                )
            if iter_num % args.n_iter_test == 0:
                total_val, pos_val, h_val, auroc = validate()
                scheduler.step(total_val)
                logger.info(
                    f'epoch={epoch} iter={iter_num} val={total_val:.4f} '
                    f'pos={pos_val:.4f} atom={h_val:.4f} auroc={auroc:.4f}'
                )
                if total_val < best_total_val:
                    best_total_val = total_val
                    save_checkpoint(
                        os.path.join(checkpoint_dir, 'best.pth'),
                        epoch, iter_num, net_dynamics, FM, optimizer, scheduler,
                    )
            if iter_num % 100 == 0:
                logger.info(
                    f'epoch={epoch} iter={iter_num}/{len(train_loader)} '
                    f'loss={float(total_loss):.4f} field={float(field_loss):.4f} '
                    f'pos={float(pos_loss):.4f} atom={float(h_loss):.4f} '
                    f'grad={float(grad_norm):.2f}'
                )
        logger.info(f'Epoch time: {time.time() - epoch_start:.1f}s')


if __name__ == '__main__':
    main()
