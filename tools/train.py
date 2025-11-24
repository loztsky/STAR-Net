"""Main script to train a model"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from starnet.utils.functions import info
from starnet.utils.build import build_model
from starnet.learners.initializer import Initializer
from starnet.learners.model import Model



def parse_json(args): # parse the json file, then add the resume and ckpt option to the cfg 
    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)
    cfg["resume"] = args.resume
    cfg["ckpt"]  = None
    cfg["TransLeanring"] = args.TransLeanring
    if args.resume or args.TransLeanring:
        cfg["ckpt"]  = args.ckpt
    return cfg


def main(args : argparse.Namespace):
    # get the json parameters
    cfg = parse_json(args)
    init = Initializer(cfg)
    data : dict = init.get_data()

    # build model
    net = build_model(cfg)

    # the information of model
    # info(net, cfg)
    
    # train
    Model(net, data).train(add_temp=cfg['add_temp'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Path to config file.',
                        default='stenet/config_files/stenet.json')
    parser.add_argument("--resume",
                        help="continue to train model?",
                        default=False)
    parser.add_argument("--TransLeanring",
                        help="transfer leanring",
                        default=False)
    parser.add_argument("--ckpt",
                        help="the path of resume check point",
                        default=None)
    parser.add_argument("--log-dir",
                        help="the dictory of log",
                        default="./logs")
    args = parser.parse_args()
    main(args)
