'''
config.py
Originally Written by Li Jiang
'''

import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    #parser.add_argument('--config', type=str, default='config/dyco3d_multigpu_scannet.yaml', help='path to config file')
    parser.add_argument('--config', type=str, default='config/dyco3d_planteye.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default="checkpoint_planteye_iter_13000.pth", help='path to pretrain model')
    #parser.add_argument('--pretrain', type=str, default="planteye_out/checkpoint_iter_8000.pth", help='path to pretrain model')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--threshold_ins', type=float, default=0.5)
    parser.add_argument('--min_pts_num', type=int, default=10)

    parser.add_argument('--resume', type=str, default="checkpoint_planteye_iter_13000.pth")
    #parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output_path', type=str, default="planteye_out")
    parser.add_argument('--use_backbone_transformer', action='store_true', default=True)


    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    assert os.path.exists(args_cfg.config)
    with open(args_cfg.config, 'r') as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


cfg = get_parser()
# setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
setattr(cfg, 'exp_path', cfg.output_path)