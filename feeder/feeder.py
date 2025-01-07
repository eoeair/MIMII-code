import torch
import random
import pickle
import numpy as np

from . import augmentations

class Feeder_snr(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path, mode):
        self.data_path = data_path
        self.mode = mode
        self.load_data()
    
    def load_data(self):
        # data: mfcc snr device label
        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)
        
        self.data = []
        self.label = []

        for i,snr in enumerate(data):
            for j in snr:
                self.data.append(j)
                self.label.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get data label
        data = self.data[index][0]
        label = self.label[index]
        
        if self.mode == 'single':
            return data, label
        elif self.mode == 'double':
            # augmentation
            data1 = self._augs(data)
            data2 = self._augs(data)
            return [data1, data2], label

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

class Feeder_device(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path, snr):
        self.data_path = data_path
        self.snr = snr
        self.load_data()
    
    def load_data(self):
        # data: mfcc snr device label
        with open(self.data_path, 'rb') as file:
            self.data = pickle.load(file)
        
    def __len__(self):
        return len(self.data[self.snr])

    def __getitem__(self, index):
        # get data
        return self.data[self.snr][index][0], self.data[self.snr][index][2]

class Feeder_label(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path, snr):
        self.data_path = data_path
        self.snr = snr
        self.load_data()
    
    def load_data(self):
        # data: mfcc snr device label
        with open(self.data_path, 'rb') as file:
            self.data = pickle.load(file)
        
    def __len__(self):
        return len(self.data[self.snr])

    def __getitem__(self, index):
        # get data
        return self.data[self.snr][index][0], self.data[self.snr][index][2]