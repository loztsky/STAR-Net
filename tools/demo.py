"""Main script to test a pretrained model"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from thop import profile
from torch import nn
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from starnet.utils.build import build_model
from matplotlib import pyplot as plt
from pathlib import Path
from starnet.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives
import cv2
def parse_json(args): # parse the json file, then add the resume and ckpt option to the cfg 
    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)
    cfg["input_dir"] = args.input_dir
    cfg["ckpt"] = args.ckpt
    cfg["output_dir"] = args.output_dir
    cfg["gt_dir"] = args.gt_dir
    cfg["verbose"] = args.verbose
    cfg["posfix"] = args.posfix
    return cfg

class PCTE:
    """
    Randomly horizontal flip the matrix with a proba p
    """

    def __init__(self, gamma=1.5):
        self.gamma = gamma

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=1).copy()
        rd_amp = np.abs(matrix)
        rd_db = 20 * np.log10(rd_amp + np.finfo(float).eps)

        # Step 2: Percentile-based dynamic range truncation
        max_db = np.percentile(rd_db, 99.5)
        min_db = np.percentile(rd_db, 20)
        min_db = min(min_db, max_db - np.finfo(float).eps)  # Ensure max > min

        # Step 3: Normalization
        i_norm = (rd_db - min_db) / (max_db - min_db)
        i_norm = np.clip(i_norm, 0, 1)

        # Step 4: Gamma correction
        matrix = np.power(i_norm, self.gamma)
        return {'matrix': matrix, 'mask': mask}

def load_data_file(file_path, file_extension):
    """
    根据文件扩展名加载不同格式的数据文件
    
    Args:
        file_path: 文件路径
        file_extension: 文件扩展名 ('.npy' 或 '.png')
    
    Returns:
        numpy数组格式的数据
    """
    if file_extension == '.npy':
        # 加载npy文件
        return np.load(file_path)
    elif file_extension == '.png':
        # 加载PNG文件
        # 使用cv2加载为灰度图像，保持数值精度
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法加载PNG文件: {file_path}")
        
        # 如果是彩色图像，转换为灰度
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 转换为float32并归一化到合适的范围
        # 假设PNG图像的像素值在0-255范围内
        return image.astype(np.float32)
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")
def load_inputdir(cfg):
    
    # pcte = PCTE() # 

    data_names = os.listdir(Path(cfg["input_dir"]))
    
    data_paths = [os.path.join(Path(cfg["input_dir"]),name) for name in data_names] # the path of input data
    datanum = len(data_paths) - cfg["nb_input_channels"] # for the num
    rdms = []
    rds = []
    print("load the input data")
    for i in tqdm(range(datanum)):
        rd_matrices = []
        for j in range(cfg["nb_input_channels"]):
            # print(f"load rd data : {data_paths[i + j]}")
            rd_matrix = load_data_file(data_paths[i + j], cfg["posfix"])
            rd_matrices.append(rd_matrix)

        rd_matrix = np.dstack(rd_matrices) # 堆叠在一起
        rd_matrix = np.rollaxis(rd_matrix, axis=-1) # 旋转
        rdm = torch.from_numpy(rd_matrix).cuda()
        rdm = normalize(rdm, 'range_doppler', norm_type=cfg["norm_type"])
        rdms.append(rdm)
        rds.append(rd_matrices[-1])
    return rdms, rds

def load_gtdir(cfg):
    initnum = cfg["nb_input_channels"]
    data_names = os.listdir(Path(cfg["gt_dir"]))
    data_paths = [os.path.join(Path(cfg["gt_dir"]), name) for name in data_names] # the path of input data
    gtnum = len(data_paths) - initnum
    gt_masks = []

    print("load ground truth masks")
    for i in tqdm(range(gtnum)):
        if cfg["posfix"] == ".png":
            datapath = os.path.join(cfg["gt_dir"], data_paths[i + initnum])
            # gt_masks.append(np.load(os.path.join(cfg["gt_dir"], data_paths[i + initnum])))
        elif cfg["posfix"] == ".npy":
            datapath = os.path.join(cfg["gt_dir"], data_paths[i + initnum], "range_doppler.npy")
            # gt_masks.append(np.load(os.path.join(cfg["gt_dir"], data_paths[i + initnum], "range_doppler.npy")))
        gt_masks.append(load_data_file(datapath, cfg["posfix"]))

    return gt_masks

# todo
def main(args : argparse.Namespace):
    cfg = parse_json(args)
    # for output dir
    output_dir = Path(args.output_dir)
    if args.only_tar:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("build model!")
    model = build_model(cfg)
    ckpt = torch.load(cfg["ckpt"])
    if ckpt["state_dict"]:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.cuda().eval()

    print("get the input!")
    input = load_inputdir(cfg)
    rdms, rds = input
    initnum = cfg["nb_input_channels"]


    if args.gt_dir:
        print("get the groundtruth masks")
        gt_masks = load_gtdir(cfg)
        height = 2
        width = 2
    else:
        gt_masks = rdms
        height = 1
        width = 3
        print("without the ground truth mask")
    
    model.eval()
    with torch.no_grad():
        for i, (rdm, rd, gt_mask) in enumerate(zip(rdms, rds, gt_masks)):
            rdm = rdm.unsqueeze(dim=0).to(torch.float32)
            if cfg["add_temp"]:
                    rdm = rdm.unsqueeze(dim=0).to(torch.float32)
            print(f"inference the {i + initnum}th radar echo!")
            # 逻辑值
            logit = model(rdm)

            # 概率图
            confidence_map = F.softmax(logit, dim=1)

            # 转换为numpy格式
            confidence_map = confidence_map.detach().cpu().numpy().squeeze()
            
            # 掩码
            mask = np.argmax(confidence_map, axis=0)

            # 创建彩色图像，背景为黑色，目标为绿色
            h, w = mask.shape
            colored_target = np.zeros((h, w, 3), dtype=np.uint8)  # 创建黑色背景
            colored_target[mask == 1] = [0, 0, 255]  # 将海杂波区域设置为绿色
            colored_target[mask == 2] = [0, 255, 0]  # 将目标区域设置为绿色


            if args.only_tar:
                # plt.imsave(os.path.join(output_dir, f"ARDtgt-{(i + initnum):06d}.png"), colored_target)
                output_file = os.path.join(output_dir,  f"ARDtgt-{(i + initnum):06d}.png")
                cv2.imwrite(output_file, colored_target)
            else:
                plt.imsave(os.path.join(str(output_dir / "tars"), f"ARDtgt-{(i + initnum):06d}.png"), colored_target)
                plt.imsave(os.path.join(str(output_dir / "masks"), f"ARDtgt-{(i + initnum):06d}.png"), mask)


            if cfg["verbose"]:
                # cv2.imshow("picure", rd.astype(np.uint8))
                # cv2.waitKey()
                plt.figure(figsize=(12, 4))  # 设置整个画布大小
                plt.subplot(height, width, 1)
                plt.imshow(rd)
                plt.title("rd map")
                plt.axis('off')
                
                plt.subplot(height, width, 2)
                plt.imshow(confidence_map.transpose(1, 2, 0))
                plt.title("confidence map")
                plt.axis('off')
                
                plt.subplot(height, width, 3)
                plt.imshow(mask)
                plt.title("output mask")
                plt.axis('off')
                if args.gt_dir and cfg["posfix"] == ".npy":
                    plt.subplot(height, width, 4)
                    plt.imshow(np.argmax(gt_mask, axis=0))
                    plt.title("gt mask")
                    plt.axis('off')
                elif args.gt_dir and cfg["posfix"] == ".png":
                    plt.subplot(height, width, 4)
                    plt.imshow(gt_mask)
                    plt.title("gt mask")
                    plt.axis('off')
                plt.tight_layout()  # 自动调整子图间距
                # plt.savefig(os.path.join(str(output_dir / "figs"), f"{(i + initnum):06d}.png"))
                plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', 
                        help='Path to config file.',
                        default='logs/othr/starnet/starnet_e450_lr0.0001_s42_3/config.json')
    parser.add_argument("--ckpt",
                        help="the check point",
                        default="logs/othr/starnet/starnet_e450_lr0.0001_s42_3/results/best_state.pt")
    parser.add_argument("--input-dir",
                        help="the dirctory of input image",
                        default="/mnt/d/othr/FRDSequence/othr/dataset/FTD_A_057t/C/FRDtgt/Ka_18")
    parser.add_argument("--gt-dir",
                        help="the path of ground truth",
                        default=None)
    parser.add_argument("--output-dir",
                        help="the dircotry of output",
                        default="./output")
    parser.add_argument("--verbose", "-v",
                        help="visualize the input",
                        action="store_true",
                        default=True)
    parser.add_argument("--posfix", 
                        help="the posfix of input data", 
                        default=".png",
                        choices=[".npy", ".png"])
    parser.add_argument("--only_tar",
                        help="the flag of save the segmentic",
                        default=True)
    args = parser.parse_args()
    main(args)
