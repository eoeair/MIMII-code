import numpy as np
from scipy.interpolate import interp1d

def add_noise(mfcc_data, noise_factor=0.01):
    """ 向 MFCC 数据添加高斯噪声 """
    noise = np.random.normal(0, noise_factor, mfcc_data.shape)
    return mfcc_data + noise

def spectral_distortion(mfcc_data, distortion_factor=0.02):
    """ 对 MFCC 频谱添加扰动 """
    distortion = np.random.normal(0, distortion_factor, mfcc_data.shape)
    return mfcc_data + distortion

def time_crop(mfcc_data, crop_size=128):
    """ 随机裁剪时间维度的片段 """
    _, T = mfcc_data.shape
    if T > crop_size:
        start = np.random.randint(0, T - crop_size)
        return mfcc_data[:, start:start + crop_size]
    else:
        return mfcc_data

def time_shift(mfcc_data, shift_max=5):
    """ 随机平移时间轴 """
    shift = np.random.randint(-shift_max, shift_max)
    if shift > 0:
        return np.pad(mfcc_data[:, :-shift], ((0, 0), (shift, 0)), mode='constant')
    elif shift < 0:
        return np.pad(mfcc_data[:, -shift:], ((0, 0), (0, -shift)), mode='constant')
    else:
        return mfcc_data

def time_warp(mfcc_data, scale_range=(0.8, 1.2)):
    """ 随机拉伸或压缩时间轴 """
    C, T = mfcc_data.shape
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_T = int(T * scale)
    
    # 使用插值对时间轴进行缩放
    x = np.arange(T)
    f = interp1d(x, mfcc_data, kind='linear', axis=1, fill_value='extrapolate')
    new_x = np.linspace(0, T-1, new_T)
    return f(new_x)

def normalize(mfcc_data):
    """ 对 MFCC 数据进行标准化 """
    return (mfcc_data - np.mean(mfcc_data)) / np.std(mfcc_data)

def crop_or_pad(mfcc_data, target_length=153):
    C, T = mfcc_data.shape
    if T > target_length:
        return mfcc_data[:, :target_length]  # 裁剪到目标长度
    else:
        # 填充到目标长度
        padding = target_length - T
        return np.pad(mfcc_data, ((0, 0), (0, padding)), mode='constant', constant_values=0)
