"""Class to load the CARRADA dataset"""
import json
from starnet.utils.paths import Paths, OTHRPaths
import os
"""
用到的文件:data_seq_ref.json、light_dataset_frame_oriented.json
用到了 split 作为训练集的分割
第一层：日期目录
第二层：日期下名称
"""
class Carrada:
    """Class to load CARRADA dataset"""
    def __init__(self):
        self.paths = Paths().get()
        self.warehouse = self.paths['warehouse']
        self.carrada = self.paths['carrada']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self._split()

    def _load_data_seq_ref(self):
        path = self.carrada / 'data_seq_ref.json'
        with open(path, 'r') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.carrada / 'light_dataset_frame_oriented.json'
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        for sequence in self.annotations.keys():
            split = self.data_seq_ref[sequence]['split']
            if split == 'Train':
                self.train[sequence] = self.annotations[sequence]
            elif split == 'Validation':
                self.validation[sequence] = self.annotations[sequence]
            elif split == 'Test':
                self.test[sequence] = self.annotations[sequence]
            else:
                raise TypeError('Type {} is not supported for splits.'.format(split))

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


class OTHR:
    def __init__(self):
        self.paths = OTHRPaths().get()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
    def get(self, dataset_type):
        if dataset_type == "Train":
            seq_names = os.listdir(self.paths['othr'] / "train")
            for seq_name in seq_names:
                self.train[seq_name] = os.listdir(str(self.paths['othr'] / "train" / seq_name / "range_doppler_processed"))
            return self.train
        elif dataset_type== "Validation":
            seq_names = os.listdir(self.paths['othr'] / "val")
            for seq_name in seq_names:
                self.validation[seq_name] = os.listdir(self.paths['othr'] / "val" / seq_name / "range_doppler_processed")
            return self.validation
        elif dataset_type == 'Test':
            seq_names = os.listdir(self.paths['othr'] / "test")
            for seq_name in seq_names:
                self.test[seq_name] = os.listdir(self.paths['othr'] / "test" / seq_name / "range_doppler_processed")
            return self.test
        else:
            raise TypeError('Type {} is not supported for splits.'.format(dataset_type))


def testCarrada():
    """Method to test the dataset"""
    dataset = Carrada().get('Train')
    assert '2019-09-16-12-52-12' in dataset.keys()
    assert '2020-02-28-13-05-44' in dataset.keys()

def testOTHR():
    # """Method to test the dataset"""
    dataset = OTHR().get('Train')
    for seq_name in dataset:
        print(seq_name)
        print(dataset[seq_name])
    # assert '2019-09-16-12-52-12' in dataset.keys()
    # assert '2020-02-28-13-05-44' in dataset.keys()


if __name__ == '__main__':
    # testCarrada()
    testOTHR()
