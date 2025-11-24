"""Initializer class to prepare training"""
import json
from torch.utils.data import DataLoader

from starnet.utils.paths import Paths,OTHRPaths
from starnet.loaders.dataset import Carrada,OTHR
from starnet.loaders.dataloaders import SequenceCarradaDataset


class Initializer:
    """Class to prepare training model

    PARAMETERS
    ----------
    cfg: dict
        Configuration file used for train/test
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg["dataset"] == "carrada":
            self.paths = Paths().get()
        elif self.cfg["dataset"]=="othr":
            self.paths = OTHRPaths().get()

    def _getCarradadatasets(self): # SequenceCarradaDataset 只是为了得到数据集在日期目录下的数据名称
        data = [Carrada().get("Train"), Carrada().get("Validation"), Carrada().get("Test")]
        trainset = SequenceCarradaDataset(data[0])
        valset = SequenceCarradaDataset(data[1])
        testset = SequenceCarradaDataset(data[2])
        return [trainset, valset, testset]
    def _getMultiRDdatasets(self):
        data = [OTHR().get("Train"), OTHR().get("Validation"), OTHR().get("Test")]
        trainset = SequenceCarradaDataset(data[0])
        valset = SequenceCarradaDataset(data[1])
        testset = SequenceCarradaDataset(data[2])
        return [trainset, valset, testset]
    
    def _getOTHR(self):
        trainset, valset, testset = self._getMultiRDdatasets()
        trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0) # dataloader进行封装
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        return [trainloader, valloader, testloader]
    def _get_dataloaders(self):
        trainset, valset, testset = self._getCarradadatasets()
        trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0) # dataloader进行封装
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        return [trainloader, valloader, testloader]

    def _structure_data(self):
        data = dict()

        # 数据集
        if self.cfg['dataset'] == "othr":
            dataloaders = self._getOTHR()
        if self.cfg['dataset'] == "carrada":
            dataloaders = self._get_dataloaders()
        
        # 日志和结果保存
        name_exp = (self.cfg['model'] + '_' +
                    'e' + str(self.cfg['nb_epochs']) + '_' +
                    'lr' + str(self.cfg['lr']) + '_' +
                    's' + str(self.cfg['torch_seed']))
        self.cfg['name_exp'] = name_exp
        folder_path = self.paths['logs'] / self.cfg['dataset'] / self.cfg['model'] / name_exp

        temp_folder_name = folder_path.name + '_' + str(self.cfg['version'])
        temp_folder_path = folder_path.parent / temp_folder_name
       
        # 日志命名，${model}_e${epochs}_lr${lr}_s${torchseed}_${times}, 这里的time是看你训练了好几次,默认开始为 0 
        while temp_folder_path.exists():
            self.cfg['version'] += 1
            temp_folder_name = folder_path.name + '_' + str(self.cfg['version'])
            temp_folder_path = folder_path.parent / temp_folder_name
        folder_path = temp_folder_path

        # 结果路径
        self.paths['results'] = folder_path / 'results'
        self.paths['writer'] = folder_path / 'boards'
        self.paths['results'].mkdir(parents=True, exist_ok=True)
        self.paths['writer'].mkdir(parents=True, exist_ok=True)

        config_path = folder_path / 'config.json'
        with open(config_path, 'w') as fp:
            json.dump(self.cfg, fp)

        data['cfg'] = self.cfg
        data['paths'] = self.paths
        data['dataloaders'] = dataloaders
        return data


    def get_data(self):
        """Return parameters of the training"""
        return self._structure_data()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.',
                        default='/root/code/python/RadarDetection/PKC/mvrss/config_files/test.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    init = Initializer(cfg)
    data = init.get_data() #  Carrada -》 SequenceCarradaDataset -》 
    train,val,test = data["dataloaders"]
    for seq_name, data in train:
        print(seq_name,": \n",data)