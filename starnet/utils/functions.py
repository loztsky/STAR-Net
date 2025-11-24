"""A lot of functions used in our pipelines"""
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from thop import profile

from starnet.utils import MVRSS_HOME

# loss
from starnet.losses.soft_dice import SoftDiceLoss
from starnet.losses.coherence import CoherenceLoss
from starnet.losses.soft_coherence import SoftCoherenceLoss
from starnet.losses.sparse_coherence import SparseCoherenceLoss
from starnet.losses.smospa_coherence import SmoSpaCoherenceLoss
from starnet.losses.distribution_coherence import DistributionCoherenceLoss
from starnet.losses.denoise_coherence import DenoiseCoherenceLoss
from starnet.losses.mvloss import MVLoss
from starnet.losses.caloss import CALoss
from starnet.losses.focalloss import FocalLoss

# augment
from starnet.loaders.dataloaders import Rescale, Flip, HFlip, VFlip, PCTE


def get_class_weights(signal_type, dataset_type):
    """Load class weights for custom loss

    PARAMETERS
    ----------
    signal_type: str
        Supported: 'range_doppler', 'range_angle'

    RETURNS
    -------
    weights: numpy array
    """
    weight_path = MVRSS_HOME / 'weights'
    if signal_type in ('range_angle'):
        file_name = f'{dataset_type}_ra_weights.json'
    elif signal_type in ('range_doppler'):
        file_name = f'{dataset_type}_rd_weights.json'
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))
    file_path = weight_path / file_name
    with open(file_path, 'r') as fp:
        weights = json.load(fp)
    
    # for carrada
    if dataset_type == "carrada":
        weights = np.array([weights['background'], weights['pedestrian'],
                        weights['cyclist'], weights['car']])
    # for custom dataset
    elif dataset_type=="othr":
        weights = np.array([weights["background"], weights["seaclutter"],
                            weights["target"]])
    weights = torch.from_numpy(weights)
    return weights

# zlw@20220302 Balance the RD and RA Losses
def get_loss_weight(signal_type):
    """Load weight for rd and ra loss
    PARAMETERS
    ----------
    signal_type: str
        Supported: 'range_doppler', 'range_angle'

    RETURNS
    -------
    weight: numpy float
    """
    if signal_type in ('range_angle'):
        weight = 2.0
    elif signal_type in ('range_doppler'):
        weight = 1.0
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))
    return weight

def transform_masks_viz(masks, nb_classes):
    """Used for visualization"""
    masks = masks.unsqueeze(1)
    masks = (masks.float()/nb_classes)
    return masks


def get_metrics(metrics, loss, losses=None):
    """Structure the metric results

    PARAMETERS
    ----------
    metrics: object
        Contains statistics recorded during inference
    loss: tensor
        Loss value
    losses: list
        List of loss values

    RETURNS
    -------
    metrics_values: dict
    """
    metrics_values = dict()
    metrics_values['loss'] = loss.item()
    if isinstance(losses, list):
        metrics_values['loss_ce'] = losses[0].item()
        metrics_values['loss_dice'] = losses[1].item()
    acc, acc_by_class = metrics.get_pixel_acc_class()  # harmonic_mean=True)
    prec, prec_by_class = metrics.get_pixel_prec_class()
    recall, recall_by_class = metrics.get_pixel_recall_class()  # harmonic_mean=True)
    miou, miou_by_class = metrics.get_miou_class()  # harmonic_mean=True)
    dice, dice_by_class = metrics.get_dice_class()
    metrics_values['acc'] = acc
    metrics_values['acc_by_class'] = acc_by_class.tolist()
    metrics_values['prec'] = prec
    metrics_values['prec_by_class'] = prec_by_class.tolist()
    metrics_values['recall'] = recall
    metrics_values['recall_by_class'] = recall_by_class.tolist()
    metrics_values['miou'] = miou
    metrics_values['miou_by_class'] = miou_by_class.tolist()
    metrics_values['dice'] = dice
    metrics_values['dice_by_class'] = dice_by_class.tolist()
    return metrics_values


def normalize(data, signal_type, norm_type='local'):
    """
    Method to normalise the radar views

    PARAMETERS
    ----------
    data: numpy array
        Radar view (batch)
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    norm_type: str
        Type of normalisation to apply
        Supported: 'local', 'tvt'

    RETURNS
    -------
    norm_data: numpy array
        normalised radar view
    """
    if norm_type in ('local'):
        min_value = torch.min(data)
        max_value = torch.max(data)
        norm_data = torch.div(torch.sub(data, min_value), torch.sub(max_value, min_value))
        return norm_data

    elif signal_type == 'range_doppler':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'weights' / 'rd_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            rd_stats = json.load(fp)
        min_value = torch.tensor(rd_stats['min_val'])
        max_value = torch.tensor(rd_stats['max_val'])

    elif signal_type == 'range_angle':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'weights' / 'ra_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ra_stats = json.load(fp)
        min_value = torch.tensor(ra_stats['min_val'])
        max_value = torch.tensor(ra_stats['max_val'])

    elif signal_type == 'angle_doppler':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'weights' / 'ad_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ad_stats = json.load(fp)
        min_value = torch.tensor(ad_stats['min_val'])
        max_value = torch.tensor(ad_stats['max_val'])

    else:
        raise TypeError('Signal {} is not supported.'.format(signal_type))

    norm_data = torch.div(torch.sub(data, min_value),
                          torch.sub(max_value, min_value))
    return norm_data


def define_loss(signal_type, custom_loss, device, dataset_type):
    """
    Method to define the loss to use during training

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' or 'angle_doppler'
    custom loss: str
        Short name of the custom loss to use
        Supported: 'wce', 'sdice', 'wce_w10sdice' or 'wce_w10sdice_w5col'
        Default: Cross Entropy is used for any other str
    devide: str
        Supported: 'cuda' or 'cpu'
    """
    if custom_loss == 'wce':
        weights = get_class_weights(signal_type, dataset_type)
        loss = nn.CrossEntropyLoss(weight = weights.to(device).float())
    elif custom_loss == 'sdice':
        loss = SoftDiceLoss()
    elif custom_loss == 'wce_w10sdice':
        weights = get_class_weights(signal_type, dataset_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, SoftDiceLoss(global_weight=10.)]
    elif custom_loss == 'wce_w10sdice_w5col':
        weights = get_class_weights(signal_type, dataset_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, SoftDiceLoss(global_weight=10.), CoherenceLoss(global_weight=5.)]
    # zlw@20220302
    elif custom_loss == 'wce_w10sdice_w5col_sig_blnc':
        weights = get_class_weights(signal_type, dataset_type)
        loss_weight = get_loss_weight(signal_type)
        weights = loss_weight * weights
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        ce_loss = ce_loss
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.*loss_weight),
                CoherenceLoss(global_weight=5.)]
    # zlw@20220304
    elif custom_loss == 'wce_w10sdice_w5sofcol':
        weights = get_class_weights(signal_type, dataset_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        ce_loss = ce_loss
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.),
                SoftCoherenceLoss(global_weight=5., relax_factor=0.2, margin=0.01)]
    # zlw@20220321
    elif custom_loss == 'wce_w10sdice_w5spacol':
        weights = get_class_weights(signal_type, dataset_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                SparseCoherenceLoss(global_weight=5.)]
    # zlw@20220322
    elif custom_loss == 'wce_w10sdice_w5smospacol':
        weights = get_class_weights(signal_type, dataset_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                SmoSpaCoherenceLoss(global_weight=5.)]
    # zlw@20220322
    elif custom_loss == 'wce_w10sdice_w5discol':
        weights = get_class_weights(signal_type, dataset_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                DistributionCoherenceLoss(global_weight=5.)]
    # zlw@20220324
    elif custom_loss == 'wce_w10sdice_w5dnscol':
        weights = get_class_weights(signal_type, dataset_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                DenoiseCoherenceLoss(global_weight=5.)]
    # for transradar
    elif custom_loss == 'ca_w10sdice_mv':
        ca_loss = CALoss(delta = 0.6, global_weight = 1., device = device)
        loss = [ca_loss, 
                SoftDiceLoss(global_weight=10.), 
                MVLoss(global_weight=5.)
                ]
    else:
        loss = nn.CrossEntropyLoss()
    return loss

def define_lossnew(signal_type, custom_loss, device, dataset_type):
    """
    Method to define the loss to use during training

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' or 'angle_doppler'
    custom loss: str
        Short name of the custom loss to use
        Supported: 'wce', 'sdice', 'wce_w10sdice' or 'wce_w10sdice_w5col'
        Default: Cross Entropy is used for any other str
    devide: str
        Supported: 'cuda' or 'cpu'
    """
    losses = []
    # single view
    if 'wce' in custom_loss:
        weights = get_class_weights(signal_type, dataset_type)
        losses.append(nn.CrossEntropyLoss(weight = weights.to(device).float()))

    
    # for transradar
    if 'ca' in custom_loss:
        if 'focal' in custom_loss:
            weights = get_class_weights(signal_type, dataset_type)
            losses.append(FocalLoss(gamma = 2, weight = weights))
        else:
            losses.append(CALoss(delta = 0.6, global_weight = 1., device = device))
    
    # softdice loss
    if 'sdice' in custom_loss:
        if 'w10sdice' in custom_loss:
            losses.append(SoftDiceLoss(global_weight=10.))
        else:
            losses.append(SoftDiceLoss())


    # coherence
    if 'w5col' in custom_loss:
        losses.append(CoherenceLoss(global_weight=5.))
    
    # # zlw@20220302
    # if custom_loss == 'wce_w10sdice_w5col_sig_blnc':
    #     weights = get_class_weights(signal_type, dataset_type)
    #     loss_weight = get_loss_weight(signal_type)
    #     weights = loss_weight * weights
    #     ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
    #     ce_loss = ce_loss
    #     loss = [ce_loss, 
    #             SoftDiceLoss(global_weight=10.*loss_weight),
    #             CoherenceLoss(global_weight=5.)]
    # zlw@20220304
    if 'w5sofcol' in custom_loss:
        losses.append(SoftCoherenceLoss(global_weight=5., relax_factor=0.2, margin=0.01))
    # zlw@20220321
    if 'w5spacol' in custom_loss:
        losses.append(SparseCoherenceLoss(global_weight=5.))
    # zlw@20220322
    if 'w5smospacol' in custom_loss:
        losses.append(SmoSpaCoherenceLoss(global_weight=5.))
    # zlw@20220322
    if 'w5discol' in custom_loss:
        losses.append(DistributionCoherenceLoss(global_weight=5.))
    # zlw@20220324
    if 'w5dnscol' in custom_loss:
        losses.append(DenoiseCoherenceLoss(global_weight=5.))
    # for transradar
    if 'mv' in custom_loss:
        losses.append(MVLoss(global_weight=5.))
    
    
    print("Signal type : ", signal_type)
    # 打印一下
    for i, loss in enumerate(losses):
        print(f"the {i + 1}th loss : ",loss)
    return losses




def get_transformations(transform_names, split='train', sizes=None):
    """Create a list of functions used for preprocessing

    PARAMETERS
    ----------
    transform_names: list
        List of str, one for each transformation
    split: str
        Split currently used
    sizes: int or tuple (optional)
        Used for rescaling
        Default: None
    """
    transformations = list()
    if 'rescale' in transform_names:
        transformations.append(Rescale(sizes))
    if 'flip' in transform_names and split == 'train':
        transformations.append(Flip(0.5))
    if 'vflip' in transform_names and split == 'train':
        transformations.append(VFlip())
    if 'hflip' in transform_names and split == 'train':
        transformations.append(HFlip())
    if 'pcte' in transform_names and split == 'train':
        transformations.append(PCTE())
    return transformations


def mask_to_img(mask):
    """Generate colors per class, only 3 classes are supported"""
    mask_img = np.zeros((mask.shape[0],
                         mask.shape[1], 3), dtype=np.uint8)
    mask_img[mask == 1] = [255, 0, 0] # 海杂波
    mask_img[mask == 2] = [0, 255, 0] # 目标
    mask_img[mask == 3] = [0, 0, 255]
    mask_img = Image.fromarray(mask_img)
    return mask_img


def get_qualitatives(outputs, masks, paths, seq_name, quali_iter, signal_type=None):
    """
    Method to get qualitative results

    PARAMETERS
    ----------
    outputs: torch tensor
        Predicted masks
    masks: torch tensor
        Ground truth masks
    paths: dict
    seq_name: str
    quali_iter: int
        Current iteration on the dataset
    signal_type: str

    RETURNS
    -------
    quali_iter: int
    """
    if signal_type:
        folder_path = paths['logs'] / signal_type / seq_name[0]
    else:
        folder_path = paths['logs'] / seq_name[0]
    folder_path.mkdir(parents=True, exist_ok=True)
    outputs = torch.argmax(outputs, axis=1).cpu().numpy()
    masks = torch.argmax(masks, axis=1).cpu().numpy()
    for i in range(outputs.shape[0]):
        mask_img = mask_to_img(masks[i])
        mask_path = folder_path / 'mask_{}.png'.format(quali_iter)
        mask_img.save(mask_path)
        output_img = mask_to_img(outputs[i])
        output_path = folder_path / 'output_{}.png'.format(quali_iter)
        output_img.save(output_path)
        quali_iter += 1
    return quali_iter


def count_params(model):
    """Count trainable parameters of a PyTorch Model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    return nb_params

def info(model, cfg):
    print("model : ", cfg["model"])
    print("dataset type : ",cfg["dataset"])
    # 参数量
    params = sum(p.numel() for p in model.parameters())
    
    # FLOPs
    if cfg["dataset"] == "carrada":
        input_rd = torch.rand((1, 1, cfg['nb_input_channels'], 256, 64)) # carrada数据
    elif cfg["dataset"] == "othr":
        input_rd = torch.rand((1, 1, cfg['nb_input_channels'], 512, 512)) # othr 数据
    input_ra = torch.rand((1, 1, cfg['nb_input_channels'], 256, 256))
    input_ad = torch.rand((1, 1, cfg['nb_input_channels'], 256, 64))
    
    if cfg['n_Input'] == 1:
        flops, _ = profile(model, inputs=(input_rd))
    elif (not cfg['add_temp'] )and cfg['n_Input'] == 2:
        input_rd = torch.squeeze(input_rd, 1)
        input_ra = torch.squeeze(input_ra, 1)
        flops, _ = profile(model, inputs=(input_rd, input_ra)) 
    elif (not cfg['add_temp'] )and cfg['n_Input'] == 3:
        input_rd = torch.squeeze(input_rd, 1)
        input_ra = torch.squeeze(input_ra, 1)
        input_ad = torch.squeeze(input_ad, 1)
        flops, _ = profile(model, inputs=(input_rd, input_ra, input_ad)) 
    else:
        flops, _ = profile(model, inputs=(input_rd, input_ra, input_ad))
    # 层数
    print(f"参数量: {params / 1e6:.2f} M, FLOPs : {flops / 1e9:.2f} G")