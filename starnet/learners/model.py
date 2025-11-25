"""Class to train a PyTorch model"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts, StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import functional as F


from starnet.loaders.dataloaders import CarradaDataset,OTHRDataset
from starnet.learners.tester import Tester
from starnet.utils.functions import normalize, define_loss, get_transformations, define_lossnew
from starnet.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer
from starnet.utils.build import build_optimizer, build_scheduler
from starnet.utils.logger import logger


from time import time
from datetime import datetime







class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
    """

    def __init__(self, net : nn.Module , data : dict):
        super().__init__()
        self.net = net
        self.parser_config(data)

    def parser_config(self, data):
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.model_name = self.cfg['model']
        self.resume = self.cfg["resume"]
        self.TransLeanring = self.cfg["TransLeanring"]
        self.ckpt = self.cfg["ckpt"]
        self.process_signal = self.cfg['process_signal']
        self.annot_type = self.cfg['annot_type']
        self.schedular_type = self.cfg['schedular']
        self.optimizer_type = self.cfg['optimizer']
        # self.w_size = self.cfg['w_size']
        # self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.loss_step = self.cfg['loss_step']
        self.val_epoch = self.cfg['val_epoch']
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.writer = SummaryWriter(self.paths['writer'])
        self.visualizer = TensorboardMultiLossVisualizer(self.writer)
        self.tester = Tester(self.cfg, self.visualizer)
        self.results = dict()

        # 针对 rodnet 系列的标志
        self.sqflag = True if "rodnet" in self.cfg["model"] else False
    
    def train(self, add_temp=False):
        """
        Method to train a network

        PARAMETERS
        ----------
        add_temp: boolean
            Add a temporal dimension during training?
            Considering the input as a sequence.
            Default: False
            简单来说 add_temp = True 的时候就是 N x D x C x H x W, add_temp = False 的时候就是 N x C x H x W
        """
        # 预备设置，包括损失日志、数据集、数据转换、随机种子
        self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names)
        self._set_seeds()
        
        # 模型
        self.net.apply(self._init_weights) # 初始化权重
        logger.info(f"Model name: {self.net.__class__.__name__}")

        # 定义损失
        # rd_criterion = define_loss('range_doppler', self.custom_loss, self.device, self.cfg["dataset"]) 
        # ra_criterion = define_loss('range_angle', self.custom_loss, self.device, self.cfg["dataset"])
        rd_criterion = define_lossnew('range_doppler', self.custom_loss, self.device, self.cfg["dataset"]) 
        ra_criterion = define_lossnew('range_angle', self.custom_loss, self.device, self.cfg["dataset"])
        
        nb_losses = len(rd_criterion)
        running_losses = []
        rd_running_losses = []
        rd_running_global_losses = [[], []]
        ra_running_losses = []
        ra_running_global_losses = [[], []]
        coherence_running_losses = []

        # 优化器，采用 Adam
        optimizer = build_optimizer(self.optimizer_type, self.net, self.lr)
        
        # 打印优化器信息
        logger.info(f"Optimizer: {optimizer.__class__.__name__}")
        logger.info(f"Optimizer hyperparameters: lr={optimizer.param_groups[0]['lr']}, weight_decay={optimizer.param_groups[0]['weight_decay']}")

        # 三种学习率调度器
        scheduler = build_scheduler(self.schedular_type, optimizer)
        logger.info(f"Scheduler: {scheduler.__class__.__name__}")


        iteration = 0
        best_val_prec = 0
        best_val_recall = 0
        best_test_dice = 0
        best_val_dop = 0
        best_test_dop = 0
        # zlw@20211012
        # if torch.cuda.device_count() > 1: # 多卡训练
        #     self.net = nn.DataParallel(self.net, device_ids=[0, 1])
        
        # 继续训练
        start_epoch = 0
        if self.resume and not self.TransLeanring:
            checkpoint = torch.load(self.ckpt, map_location=self.device)
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Move optimizer states to the correct gpu device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_val_dop = checkpoint['best_val_dop']
            best_test_dop = checkpoint['best_test_dop']
            logger.info("=> Loaded checkpoint at epoch {} w/ best dice in valuation: {}, best dice in testing: {})".format(
                checkpoint['epoch'], best_val_dop, best_test_dop))
        # 迁移学习
        if self.TransLeanring and not self.resume:
            checkpoint = torch.load(self.ckpt, map_location=self.device)
            self.net.load_state_dict(checkpoint['state_dict'], strict=False)
        self.net.to(self.device)
        logger.info("dataset type : {}".format(self.cfg["dataset"]))

        train_start = time()
        logger.info("the start time of training : {}".format(datetime.fromtimestamp(train_start)))

        for epoch in range(start_epoch, self.nb_epochs):
            epoch_start = time()
            if epoch % self.lr_step == 0 and epoch != 0: # 调整学习率
                scheduler.step()
            for sequence_data in train_loader: # 获取数据集的名称
                seq_name, seq = sequence_data
                # timepoint2 = time()
                # logger.info(f"cost time 1 : {train_start - timepoint1}")
                # 下载多帧数据集
                if self.cfg["dataset"] == "carrada":
                    path_to_frames = os.path.join(self.paths['carrada'], seq_name[0]) # 数据集序列的目录路径，例如 D:\dataset\Radar\Carrada\2019-09-16-12-52-12
                    frame_dataloader = DataLoader(CarradaDataset(seq,
                                                                self.annot_type,
                                                                path_to_frames,
                                                                self.process_signal,
                                                                self.n_frames,
                                                                transformations,
                                                                add_temp),
                                                shuffle=self.is_shuffled,
                                                batch_size=self.batch_size,
                                                num_workers=4)
                elif self.cfg["dataset"] == "othr":
                    path_to_frames = os.path.join(self.paths['othr'], "train", seq_name[0]) # 数据集序列的目录路径，例如 D:\dataset\Radar\lab\OTHR\data\train\2019-09-16-12-52-12
                    frame_dataloader = DataLoader(OTHRDataset(seq,
                                                                path_to_frames,
                                                                self.process_signal,
                                                                self.n_frames,
                                                                transformations,
                                                                add_temp),
                                                shuffle=self.is_shuffled,
                                                batch_size=self.batch_size,
                                                num_workers=4)
                # timepoint3 = time()
                # logger.info(f"read dataset time : {timepoint3 - timepoint2}")
                for _, frame in enumerate(frame_dataloader):
                    # 数据集
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    // rd_data_np = rd_data.cpu().numpy()  # 转换为 NumPy

                    # import cv2
                    # 选择 batch 中的第一帧
                    // frame_idx = 0
                    // single_frame = rd_data_np[0, frame_idx, :, :]

                    # 归一化到 [0, 255]
                    // normalized_frame = (single_frame - np.min(single_frame)) / (np.max(single_frame) - np.min(single_frame) + 1e-6)
                    // normalized_frame = (normalized_frame * 255).astype(np.uint8)

                    # OpenCV 显示
                    # cv2.imshow("RD Matrix", normalized_frame)
                    # cv2.waitKey(0)  # 按任意键关闭窗口
                    # cv2.destroyAllWindows()
                    rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                    if self.cfg["n_Input"] == 2:
                        ra_data = frame['ra_matrix'].to(self.device).float()
                        ra_mask = frame['ra_mask'].to(self.device).float()
                        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                    elif self.cfg["n_Input"] == 3:
                        ra_data = frame['ra_matrix'].to(self.device).float()
                        ra_mask = frame['ra_mask'].to(self.device).float()
                        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                        ad_data = frame['ad_matrix'].to(self.device).float()
                        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                    # timepoint4 = time()
                    optimizer.zero_grad()
                    if self.cfg["n_Input"] == 1 and self.cfg["n_Output"] == 1:
                        rd_outputs = self.net(rd_data) # to do
                        rd_outputs = rd_outputs.to(self.device)
                        if self.sqflag:
                            rd_outputs = rd_outputs.squeeze(2) 
                    elif self.cfg["n_Output"] == 2:
                        if self.cfg["n_Input"] == 2 :
                            rd_outputs, ra_outputs = self.net(rd_data, ra_data)
                        elif self.cfg["n_Input"] == 3 :
                            rd_outputs, ra_outputs = self.net(rd_data, ra_data, ad_data)
                        rd_outputs = rd_outputs.to(self.device)
                        ra_outputs = ra_outputs.to(self.device)
                        if self.sqflag:
                            rd_outputs = rd_outputs.squeeze(2)
                            ra_outputs = ra_outputs.squeeze(2) 
                    
                    # 尺寸不一致
                    if rd_outputs.shape[-2:] != rd_data.shape[-2:]:
                        rd_outputs = F.interpolate(
                            rd_outputs,
                            size=rd_data.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    # timepoint5 = time()
                    # logger.info(f"inference time : {timepoint5 - timepoint4}")
                    if self.cfg["n_Output"] == 1:
                        rd_losses = [criterion(rd_outputs, torch.argmax(rd_mask, axis=1))
                                        for criterion in rd_criterion] #  the wCE and wSDice of rd
                        rd_loss = torch.mean(torch.stack(rd_losses)) # average of wCE and wSDice loss  of rd
                        loss = torch.mean(torch.stack(rd_losses))# average of wCE and wSDice loss ,as the global loss
                    elif self.cfg["n_Output"] == 2:
                        if nb_losses < 3:
                            # Case without the CoL
                            rd_losses = [criterion(rd_outputs, torch.argmax(rd_mask, axis=1))
                                        for criterion in rd_criterion]                          # wCE and wSDice
                            rd_loss = torch.mean(torch.stack(rd_losses))                        # average of wCE and wSDice loss 
                            ra_losses = [criterion(ra_outputs, torch.argmax(ra_mask, axis=1))
                                        for criterion in ra_criterion]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            loss = torch.mean(rd_loss + ra_loss)
                        else:
                            # Case with the CoL
                            ## Select the wCE and wSDice
                            rd_losses = [criterion(rd_outputs, torch.argmax(rd_mask, axis=1))
                                        for criterion in rd_criterion[:2]]                      # wCE and wSDice
                            rd_loss = torch.mean(torch.stack(rd_losses))                        # average of wCE and wSDice loss 
                            ra_losses = [criterion(ra_outputs, torch.argmax(ra_mask, axis=1))
                                        for criterion in ra_criterion[:2]]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            
                            ## Coherence loss
                            coherence_loss = rd_criterion[2](rd_outputs, ra_outputs)
                            loss = torch.mean(rd_loss + ra_loss + coherence_loss)               # average loss
                    # timepoint6 = time()
                    # logger.info(f"loss processing time : {timepoint6 - timepoint5}")
                    # 梯度反向
                    loss.backward()
                    optimizer.step()
                    # timepoint7 = time()
                    # logger.info(f"back time : {timepoint7 - timepoint6}")
                    
                    # running_losses.append(loss.data.cpu().numpy()[()])
                    # rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                    # rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                    # rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                    running_losses.append(loss.item())
                    rd_running_losses.append(rd_loss.item())
                    for i in range(2):
                        rd_running_global_losses[i].append(rd_losses[i].item())
                    # timepoint8 = time()
                    # logger.info(f"loss append time : {timepoint8 - timepoint7}")
                    if self.cfg["n_Output"] == 2:
                        # ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                        # ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                        # ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                        ra_running_losses.append(ra_loss.item())
                        for i in range(2):
                            ra_running_global_losses[i].append(ra_losses[i].item())
                    if nb_losses > 2:
                        coherence_running_losses.append(coherence_loss.item())
                    
                    # timepoint9 = time()
                    # logger.info(f"rd ra coherence time : {timepoint9 - timepoint8}")

                    if iteration % self.loss_step == 0:
                        train_loss = np.mean(running_losses)
                        rd_train_loss = np.mean(rd_running_losses)
                        rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                        if self.cfg["n_Output"] == 2:
                            ra_train_loss = np.mean(ra_running_losses)
                            ra_train_losses = [np.mean(sub_loss) for sub_loss in ra_running_global_losses]
                        # zlw@20220302
                        logger.info('[Epoch {}/{}, iter {}]: '
                              'Train loss {}'.format(epoch+1,
                                                     self.nb_epochs,
                                                     iteration,
                                                     train_loss))
                        if self.cfg["n_Output"] == 1:
                            message = '[Epoch {}/{}, iter {}]: Train losses: RD={}'.format(epoch+1,
                                                                  self.nb_epochs,
                                                                  iteration,
                                                                  rd_train_loss)
                        if self.cfg["n_Output"] == 2:
                            message = '[Epoch {}/{}, iter {}]: Train losses: RD={}, RA={}'.format(epoch+1,
                                                                  self.nb_epochs,
                                                                  iteration,
                                                                  rd_train_loss,
                                                                  ra_train_loss)
                        logger.info(message)
                        if nb_losses > 2:
                            coherence_train_loss = np.mean(coherence_running_losses)
                            # zlw@20220302
                            logger.info('[Epoch {}/{}, iter {}]: '
                                  'Train Coherence loss {}'.format(epoch+1,
                                                                   self.nb_epochs,
                                                                   iteration,
                                                                   coherence_train_loss))
                        if self.cfg["n_Output"] == 1:
                            # train_loss = np.mean(running_losses) = np.mean(list(loss.item()))
                            self.visualizer.update_single_train_loss(train_loss, rd_train_loss,
                                                                     rd_train_losses, iteration)
                        elif self.cfg["n_Output"] == 2:
                            if nb_losses > 2:
                                self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, ra_train_loss,
                                                                        ra_train_losses, iteration,
                                                                        coherence_train_loss)
                            else:
                                self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, ra_train_loss,
                                                                        ra_train_losses, iteration)
                        running_losses = list()
                        rd_running_losses = list()
                        ra_running_losses = list()
                        # zlw@20220107 get_lr() --> get_last_lr()
                        self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)
                    # timepoint10 = time()
                    # logger.info(f"loss print time : {timepoint10 - timepoint9}")
                    iteration += 1
            epoch_time = time() - epoch_start
            logger.info("the cost time the training one epoch : {} s".format(epoch_time))
            if (epoch + 1) % self.val_epoch == 0 or epoch == 0 :
                time_epoch1 = time()
                # for validation dataset
                val_metrics = self.tester.predict(self.net, val_loader, epoch, add_temp=add_temp, mode="val")
                # time_epoch2 = time()
                # logger.info(f"val metric processing time : {time_epoch2 - time_epoch1}")
                if self.cfg["n_Output"] == 1:
                    self.visualizer.update_single_val_metrics(val_metrics, epoch)
                    logger.info('[Epoch {}/{}] Val losses: '
                        'RD={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['loss']))
                    logger.info('[Epoch {}/{}] Val Pixel Prec: '
                        'RD={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['prec']))
                    logger.info('[Epoch {}/{}] Val class IoU: '
                        'RD={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['miou_by_class']))
                    logger.info('[Epoch {}/{}] Val mIoU: '
                        'RD={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['miou']))
                    logger.info('[Epoch {}/{}] Val Dice: '
                        'RD={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['dice']))
                elif self.cfg["n_Output"] == 2:
                    self.visualizer.update_multi_val_metrics(val_metrics, epoch)
                    logger.info('[Epoch {}/{}] Val losses: '
                        'RD={}, RA={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['loss'],
                                                val_metrics['range_angle']['loss'],))
                    if nb_losses > 2:
                        logger.info('[Epoch {}/{}] Val Coherence Loss: '
                        'CoL={}'.format(epoch+1,
                                        self.nb_epochs,
                                        val_metrics['coherence_loss']))
                    logger.info('[Epoch {}/{}] Val Pixel Prec: '
                        'RD={}, RA={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['prec'],
                                                val_metrics['range_angle']['prec']))
                    logger.info('[Epoch {}/{}] Val class IoU: '
                        'RD={}, RA={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['miou_by_class'],
                                                val_metrics['range_angle']['miou_by_class']))
                    logger.info('[Epoch {}/{}] Val mIoU: '
                        'RD={}, RA={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['miou'],
                                                val_metrics['range_angle']['miou']))
                    logger.info('[Epoch {}/{}] Val Dice: '
                        'RD={}, RA={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['range_doppler']['dice'],
                                                val_metrics['range_angle']['dice']))

                '''if val_metrics['global_prec'] > best_val_prec and iteration > 0:
                    best_val_prec = val_metrics['global_prec']
                '''
                if val_metrics['range_doppler']['dice'] > best_val_dop: # record best val metric
                    best_val_dop = val_metrics['range_doppler']['dice']
                
                # time_epoch3 = time()
                # logger.info(f"val message print time : {time_epoch3 - time_epoch2}")
                
                # for test dataset
                # saving results w/ best dice zlw@20220704
                test_metrics = self.tester.predict(self.net, test_loader, add_temp=add_temp, mode="test")
                # time_epoch4 = time()
                # logger.info(f"test infer time : {time_epoch4 - time_epoch3}")
                if test_metrics['global_dice'] > best_test_dice:
                    best_test_dice = test_metrics['global_dice']
                    if self.cfg["n_Output"] == 1:
                        logger.info('[Epoch {}/{}] Test losses: '
                            'RD={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['loss']))
                        logger.info('[Epoch {}/{}] Test Prec: '
                            'RD={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['prec']))
                        logger.info('[Epoch {}/{}] Test class IoU: '
                            'RD={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['miou_by_class']))
                        # zlw@20220227
                        logger.info('[Epoch {}/{}] Test mIoU: '
                            'RD={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['miou']))
                        logger.info('[Epoch {}/{}] Test Dice: '
                            'RD={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['dice']))
                    elif self.cfg["n_Output"] == 2:
                        logger.info('[Epoch {}/{}] Test losses: '
                            'RD={}, RA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['loss'],
                                                    test_metrics['range_angle']['loss']))
                        logger.info('[Epoch {}/{}] Test Prec: '
                            'RD={}, RA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['prec'],
                                                    test_metrics['range_angle']['prec']))
                        logger.info('[Epoch {}/{}] Test class IoU: '
                        'RD={}, RA={}'.format(epoch+1,
                                                self.nb_epochs,
                                                test_metrics['range_doppler']['miou_by_class'],
                                                test_metrics['range_angle']['miou_by_class']))
                        # zlw@20220227
                        logger.info('[Epoch {}/{}] Test mIoU: '
                            'RD={}, RA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['miou'],
                                                    test_metrics['range_angle']['miou']))
                        logger.info('[Epoch {}/{}] Test Dice: '
                            'RD={}, RA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    test_metrics['range_doppler']['dice'],
                                                    test_metrics['range_angle']['dice']))
                    if test_metrics['range_doppler']['dice'] > best_test_dop : # record best test metric
                            best_test_dop = test_metrics['range_doppler']['dice']
                    self.results['epoch'] = epoch + 1
                    self.results['rd_train_loss'] = rd_train_loss.item()
                    if self.cfg["n_Output"] == 2:
                        self.results['ra_train_loss'] = ra_train_loss.item()
                    self.results['train_loss'] = train_loss.item()
                    self.results['val_metrics'] = val_metrics
                    self.results['test_metrics'] = test_metrics
                    # for more infomation
                    self.state = {
                                'epoch': epoch + 1,
                                'state_dict': self.net.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'best_val_dop': best_val_dop,
                                'best_test_dop': best_test_dop,
                            }
                    if nb_losses > 3:
                        self.results['coherence_train_loss'] = coherence_train_loss.item()
                    self._save_results() # save for best test metric
                    # time_epoch5 = time()
                    # logger.info(f"test cost time : {time_epoch5 - time_epoch4}")
                    self.net.train()  # Train mode after evaluation process
            # last epoch saving for resuming training
            self.laststate = {
                                'epoch': epoch + 1,
                                'state_dict': self.net.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'best_val_dop': best_val_dop,
                                'best_test_dop': best_test_dop,
                            }
            self._save_results_laststate()
        
        self.writer.close()
        train_end = time()
        logger.info("the end time of training : {}".format(datetime.fromtimestamp(train_end)))
        logger.info("the cost time of traing : {}".format(train_end - train_start))



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:  # 检查 bias 是否存在
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:  # 检查 bias 是否存在
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):  # 修正缩进，确保是独立判断
            nn.init.uniform_(m.weight, 0., 1.)
            if m.bias is not None:  # 检查 bias 是否存在
                nn.init.constant_(m.bias, 0.)
    
    def _save_results(self):
        results_path = self.paths['results'] / 'best_results.json'
        model_path = self.paths['results'] / 'best_model.pt'
        state_path = self.paths['results'] / 'best_state.pt'
        with open(results_path, "w") as fp:
            json.dump(self.results, fp)
        torch.save(self.net.state_dict(), model_path)
        torch.save(self.state, state_path)
    
    def _save_results_laststate(self):
        laststate_path = self.paths['results'] / 'last_state.pt'
        torch.save(self.laststate, laststate_path)

    def _set_seeds(self):
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
