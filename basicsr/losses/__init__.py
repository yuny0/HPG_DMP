from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss, WeightedTVLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty)


def build_loss(opt):

    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
