'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default="config/pointgroup_default_planteye.yaml", help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str,
                        default='pretrain_planteye.pth',
                        help='path to pretrain model')

    args_cfg = parser.parse_args()
    print(args_cfg)
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg



def get_parser_nb():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    args_cfg = parser.parse_args(args=[])
    #args_cfg = parser.parse_args()
    print(args_cfg)
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg



cfg = get_parser()
#cfg =get_parser_nb()
#setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
