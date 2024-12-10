import torch
import pickle
import numpy as np

class Feeder(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        # data: mfcc snr device label
        with open(self.data_path, 'rb') as file:
            self.data = pickle.load(file)
        

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        # get data
        return np.array(self.data[1][index][0]), self.data[1][index][2]