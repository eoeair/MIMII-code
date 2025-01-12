import torch
import random
import pickle
import numpy as np

from . import augmentations

class Feeder_base(torch.utils.data.Dataset):
    """ Feeder base """

    def __init__(self, data_path, mode, snr=0):
        self.data_path = data_path
        self.mode = mode
        self.snr = snr
        self.load_data()
    
    def load_data(self):
        # data: [SNR[MFCC,device,label]]
        pass

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data label
        data = self.data[index]
        label = self.label[index]
        
        if self.mode == 'single':
            return data, label
        elif self.mode == 'double':
            # augmentation
            data1 = self._augs(data)
            data2 = self._augs(data)
            return [data1, data2]

    def _augs(self, data):
        if random.random() < 0.5:
            data = augmentations.add_noise(data)
        if random.random() < 0.5:
            data = augmentations.spectral_distortion(data)
        if random.random() < 0.5:
            data = augmentations.time_shift(data)
        if random.random() < 0.5:
            data = augmentations.time_crop(data)
        if random.random() < 0.5:
            data = augmentations.time_warp(data)
        return augmentations.crop_or_pad(data)

class Feeder_snr(Feeder_base):
    """ Feeder for snr inputs """
    def load_data(self):

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)
        
        self.data = []
        self.label = []

        for i,snr in enumerate(data):
            for j in snr:
                self.data.append(j[0])
                self.label.append(i)

class Feeder_device(Feeder_base):
    """ Feeder for device inputs """
    
    def load_data(self):

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)[self.snr]
        
        self.data = []
        self.label = []

        for i in data:
            self.data.append(i[0])
            self.label.append(i[1])

class Feeder_label(Feeder_base):
    """ Feeder for label inputs """

    def load_data(self):

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)[self.snr]
        
        self.data = []
        self.label = []

        for i in data:
            self.data.append(i[0])
            self.label.append(i[2])