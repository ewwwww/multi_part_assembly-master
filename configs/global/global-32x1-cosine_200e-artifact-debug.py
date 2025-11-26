import os
from yacs.config import CfgNode as CN
from multi_part_assembly.utils import merge_cfg

_base_ = {
    'exp': '../_base_/default_exp.py',
    'data': '../_base_/datasets/breaking_bad/artifact.py',
    'optimizer': '../_base_/schedules/adam_cosine.py',
    'model': '../_base_/models/global.py',
    'loss': '../_base_/models/loss/geometric_loss.py',
}

# Debug configs for small-scale testing
_C = CN()
_C.data = CN()
_C.data.overfit = 10  # 只使用前10个样本进行快速调试
_C.exp = CN()
_C.exp.batch_size = 4  # 减小batch size
_C.exp.num_epochs = 2  # 只训练2个epoch
_C.exp.val_every = 1  # 每个epoch都验证
_C.exp.num_workers = 0  # 调试时设为0，避免多进程问题
_C.exp.gpus = [0]  # 使用单个GPU


def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg


