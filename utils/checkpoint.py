import torch


GUIDANCE_KEYS = {
    'null',
    'mask_embed.weight',
    'embed.0.offset',
    'vina_predictor.net.0.weight',
}


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or 'net_dynamics_state_dict' not in checkpoint:
        raise ValueError(f'Unsupported checkpoint format: {path}')
    return checkpoint


def detect_model_variant(checkpoint):
    state_dict = checkpoint['net_dynamics_state_dict']
    if GUIDANCE_KEYS.issubset(state_dict):
        return 'pfm_g'
    if GUIDANCE_KEYS.intersection(state_dict):
        present = sorted(GUIDANCE_KEYS.intersection(state_dict))
        raise ValueError(f'Incomplete PFM-G checkpoint keys: {present}')
    return 'pfm'


def configure_model_for_variant(model_config, variant):
    if variant not in {'pfm', 'pfm_g'}:
        raise ValueError(f'Unknown model variant: {variant}')
    model_config.guidance_conditioning = variant == 'pfm_g'
    model_config.predict_affinity = variant == 'pfm_g'
    return model_config


def load_model_state(model, checkpoint, expected_variant=None):
    variant = detect_model_variant(checkpoint)
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f'Checkpoint is {variant}, but the sampling config requests '
            f'{expected_variant}.'
        )
    model.load_state_dict(checkpoint['net_dynamics_state_dict'], strict=True)
    return variant


def load_predictor_state(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or 'net_dynamics_state_dict' not in checkpoint:
        raise ValueError(f'Unsupported predictor checkpoint format: {path}')
    model.load_state_dict(checkpoint['net_dynamics_state_dict'], strict=True)
    return checkpoint
