"""Main script to test a pretrained model"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch import nn
from starnet.learners.initializer import Initializer
from starnet.utils.paths import Paths
from starnet.utils.functions import info
from starnet.learners.tester import Tester
# from mvrss.models import TMVANet, MVNet, MVANet, MVA_DCN, TMVA_DCN, TMVA_TDC, PKCIn, PKCOn, SVNet, SVANet
from starnet.loaders.dataset import Carrada, OTHR
from starnet.loaders.dataloaders import SequenceCarradaDataset
from starnet.utils.build import build_model

def parse_json(args): # parse the json file, then add the resume and ckpt option to the cfg 
    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)
    return cfg


def main(args : argparse.Namespace):
    # json parse
    cfg = parse_json(args)
    paths = Paths().get()
    exp_name = cfg['name_exp'] + '_' + str(cfg['version'])
    path = paths['logs'] / cfg['dataset'] / cfg['model'] / exp_name     # the folder path
    model_path = path / "results" / "last_state.pt"                          # pretrained weights
    test_results_path = path / 'results' / 'test_results.json'          # the output json path
    
    # model
    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path)["state_dict"],strict=False)
    model.cuda().eval()
    
    # val
    tester = Tester(cfg)
    if cfg["dataset"] == "carrada" :
        data = Carrada()
        test = data.get('Test')
    elif cfg["dataset"] == "othr":
        data = OTHR()
        test = data.get('Test')
    
    testset = SequenceCarradaDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False)
    tester.set_annot_type(cfg['annot_type'])
    if cfg['model'] == 'mvnet' or cfg['model'] == 'svnet':
        test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=cfg["add_temp"], mode="test")
    else:
        test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=cfg["add_temp"], mode="test")
    tester.write_params(test_results_path)
    print(f"test result saved in {test_results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', 
                        help='Path to config file.',
                        default='logs/othr/starnet/starnet_e400_lr0.0001_s42_3/config.json')
    args = parser.parse_args()
    main(args)
