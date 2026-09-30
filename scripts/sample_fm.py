import argparse
import os
import shutil
import sys

import torch
from torch.utils.data import Subset
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from egnn.DotGat import DotPre_FM
from scripts.flow_matching_utils import CFM_CD_PERTURB
from scripts.prior_distributions import PositionFeaturePrior_PFM
from scripts.sampling import sample_ode_pfm, sample_ode_pfm_g
from utils.checkpoint import (
    configure_model_for_variant,
    detect_model_variant,
    load_checkpoint,
    load_model_state,
    load_predictor_state,
)


def resolve_path(path, config_path):
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(config_path), path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--result_path', type=str, default='./outputs')
    args = parser.parse_args()

    sampling_config_path = os.path.abspath(args.config)
    config = misc.load_config(sampling_config_path)
    training_config_path = resolve_path(config.training_yml, sampling_config_path)
    training_config = misc.load_config(training_config_path)
    checkpoint_path = resolve_path(config.model.checkpoint, sampling_config_path)
    x_predictor_path = resolve_path(
        training_config.data.x_predictor,
        training_config_path,
    )
    h_predictor_path = resolve_path(
        training_config.data.h_predictor,
        training_config_path,
    )
    training_config.data.path = resolve_path(
        training_config.data.path,
        training_config_path,
    )
    training_config.data.split = resolve_path(
        training_config.data.split,
        training_config_path,
    )
    if getattr(training_config.data, 'index_path', None) is not None:
        training_config.data.index_path = resolve_path(
            training_config.data.index_path,
            training_config_path,
        )

    logger = misc.get_logger('sampling')
    logger.info(config)
    misc.seed_all(int(config.sample.seed))

    checkpoint = load_checkpoint(checkpoint_path, args.device)
    checkpoint_variant = detect_model_variant(checkpoint)
    requested_variant = str(config.model.variant).lower()
    configure_model_for_variant(training_config.model, checkpoint_variant)

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(
        training_config.data.transform.ligand_atom_mode
    )
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])
    dataset_kwargs = {'transform': transform}
    if getattr(training_config.data, 'index_path', None) is not None:
        dataset_kwargs['index_path'] = training_config.data.index_path
    dataset, subsets = get_dataset(
        config=training_config.data,
        **dataset_kwargs,
    )
    split_name = str(getattr(config.sample, 'dataset_split', 'test'))
    if split_name not in subsets:
        raise KeyError(f'Dataset split not found: {split_name}')
    sampling_set = subsets[split_name]
    max_examples = getattr(config.sample, 'max_examples', None)
    if max_examples is not None:
        max_examples = min(int(max_examples), len(sampling_set))
        sampling_set = Subset(sampling_set, range(max_examples))

    net_dynamics = DotPre_FM(training_config.model, 27, 12)
    x_bar_predictor = DotPre_FM(training_config.modelp, 27, 12)
    h_bar_predictor = DotPre_FM(training_config.modelp, 27, 12)
    load_model_state(
        net_dynamics,
        checkpoint,
        expected_variant=requested_variant,
    )
    load_predictor_state(x_bar_predictor, x_predictor_path, args.device)
    load_predictor_state(h_bar_predictor, h_predictor_path, args.device)

    fm = CFM_CD_PERTURB(training_config.train.sigma_pos).to(args.device)
    fm.load_state_dict(checkpoint['FM_state_dict'], strict=True)
    x_para = fm.x_perturb.detach()
    h_para = fm.h_perturb.detach()
    prior = PositionFeaturePrior_PFM(n_dim=3, in_node_nf=12)

    net_dynamics = net_dynamics.to(args.device).eval()
    x_bar_predictor = x_bar_predictor.to(args.device).eval()
    h_bar_predictor = h_bar_predictor.to(args.device).eval()
    logger.info(
        f'Loaded {checkpoint_variant} checkpoint: {checkpoint_path}'
    )

    if requested_variant == 'pfm':
        sampler = sample_ode_pfm
    elif requested_variant == 'pfm_g':
        sampler = sample_ode_pfm_g
    else:
        raise ValueError('model.variant must be pfm or pfm_g')

    num_molecules = int(config.sample.num_samples)
    num_steps = int(config.sample.num_steps)
    sampler_kwargs = {
        'num_mole': num_molecules,
        'num_steps': num_steps,
    }
    if requested_variant == 'pfm_g':
        sampler_kwargs.update({
            'guidance_scale': float(config.sample.guidance_scale),
            'guidance_target': float(
                getattr(config.sample, 'guidance_target', 1.0)
            ),
        })

    os.makedirs(args.result_path, exist_ok=True)
    shutil.copyfile(
        sampling_config_path,
        os.path.join(args.result_path, 'sample.yml'),
    )
    for data_id, data in enumerate(tqdm(sampling_set)):
        with torch.no_grad():
            ligand_pos, ligand_v = sampler(
                args.device,
                data,
                x_bar_predictor,
                h_bar_predictor,
                x_para,
                h_para,
                prior,
                net_dynamics,
                **sampler_kwargs,
            )
        result = {
            'data': data,
            'pred_ligand_pos': ligand_pos,
            'pred_ligand_v': ligand_v,
            'sampling_metadata': {
                'model_variant': requested_variant,
                'dataset_split': split_name,
                'num_steps': num_steps,
                'atom_count': 'empirical_pocket_size_prior',
            },
        }
        if requested_variant == 'pfm_g':
            result['sampling_metadata']['guidance_scale'] = float(
                config.sample.guidance_scale
            )
        torch.save(
            result,
            os.path.join(args.result_path, f'result_{data_id}.pt'),
        )


if __name__ == '__main__':
    main()
