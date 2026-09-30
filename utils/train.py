import torch


def get_optimizer(cfg, model):
    if cfg.type != 'adam':
        raise NotImplementedError(f'Optimizer not supported: {cfg.type}')
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )


def get_scheduler(cfg, optimizer):
    if cfg.type != 'plateau':
        raise NotImplementedError(f'Scheduler not supported: {cfg.type}')
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg.factor,
        patience=cfg.patience,
        min_lr=cfg.min_lr,
    )
