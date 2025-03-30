import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
#import util
import math
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.interpolate import interp1d
import datetime
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy import signal as signal

folder_path = 'E:\\ccdt_data'
data_path = 'ccdt_data'

testset = 0


if testset == 1:
    NAMES = ['HUP179']
else:
    #exclude_files = {'HUP139','HUP154'} #信号差的，目前还没用到

    all_files = os.listdir(os.path.join('E:/ccdt_data',data_path))
    NAMES = [file for file in all_files if file.startswith('HUP')]
    print(NAMES)

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)
    return obj

# 读取受试者EEG数据
def loadSubjData(save_dir, subj):
    """加载特定受试者的数据"""
    if os.path.exists(save_dir + subj):
        thisSubjData = load_pickle(save_dir + subj)
        eeg_data = thisSubjData[0]  # EEG数据 (numpy数组)
        eeg_metadata = thisSubjData[1]  # EEG元数据
        beh_df = thisSubjData[2]  # 行为数据
        return eeg_data, eeg_metadata, beh_df
    else:
        print('Error: Data for this subject not found')
        return None, None, None


def compute_kurtosis(eeg_data):
    """
    计算每个电极的Kurtosis（在时间维度上）。
    eeg_data: numpy数组, 形状为 (trials, time, electrodes)
    返回: (electrodes,)
    """
    num_trials, num_samples, num_channels = eeg_data.shape
    kurt_values = np.zeros(num_channels)

    for ch in range(num_channels):
        all_trials_signal = eeg_data[:, :, ch].flatten()  # 将所有trial展开为1D数组
        kurt_values[ch] = kurtosis(all_trials_signal, fisher=False)  # 计算峰度（不减3）

    return kurt_values


# 计算功率谱密度（PSD）
def compute_psd(eeg_signal, fs):
    freqs, psd = signal.welch(eeg_signal, fs=fs, nperseg=fs*2)
    return freqs, psd

def compute_snr_concatenated(eeg_data, fs):
    """
    计算 SNR，使用所有 trials 连接的方法
    eeg_data 形状为 (trials, time, electrodes)
    """
    num_trials, num_samples, num_channels = eeg_data.shape
    snr_values = np.zeros(num_channels)

    for ch in range(num_channels):
        concatenated_signal = eeg_data[:, :, ch].flatten()  # 连接所有 trials
        freqs, psd = compute_psd(concatenated_signal, fs)

        signal_band = (freqs >= 70) & (freqs <= 200)
        noise_band = (freqs < 70) | (freqs > 200)

        P_signal = np.sum(psd[signal_band])
        P_noise = np.sum(psd[noise_band])

        snr_values[ch] = 10 * np.log10(P_signal / P_noise)

    return snr_values

# 主函数
def main(NAME):
    """
    处理 EEG 数据，计算 Kurtosis、PSD 和 SNR。
    """
    print(f"Processing Subject: {NAME}")

    # 读取EEG数据
    eeg_data, eeg_metadata, beh_df = loadSubjData('E:/ccdt_data/ccdt_data/', NAME)

    if eeg_data is None:
        return
    print(eeg_data.shape)
    fs = eeg_metadata['samplerate']  # 采样率
    print(f"Sampling Rate: {fs} Hz")
    # 计算Kurtosis
    kurt_values = compute_kurtosis(eeg_data)
    print("Kurtosis finished")

    # 计算SNR
    snr_values = compute_snr_concatenated(eeg_data, fs)
    print("SNR finished")

    # 计算所有通道的 PSD
    num_channels = eeg_data.shape[2]  # 获取通道数
    psd_all_channels = []


    for ch in range(num_channels):
        num_trials, num_timepoints = eeg_data.shape[:2]

        # 处理第一个 trial
        adjusted_signal = eeg_data[0, :, ch].tolist()  # 将第一个 trial 作为起点
         # 处理后续 trials
        for t in range(1, num_trials):
            trial = eeg_data[t, :, ch]
            offset = trial[0] - adjusted_signal[-1]  # 计算偏移量
            trial_adjusted = trial - offset  # 整个 trial 平移
            adjusted_signal.extend(trial_adjusted)  # 拼接调整后的 trial
        adjusted_signal = np.array(adjusted_signal)  # 转换为 numpy 数组

        freqs, psd = compute_psd(adjusted_signal, fs)
        psd_all_channels.append(psd)


    # 计算所有通道的均值 PSD
    psd_mean = np.mean(psd_all_channels, axis=0)

    # 绘制Kurtosis分布
    plt.figure(figsize=(10, 4))
    plt.hist(kurt_values, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=10, color='r', linestyle='--', label="Kurtosis = 10")
    plt.xlabel("Kurtosis")
    plt.ylabel("Frequency")
    plt.title("Kurtosis distribution of EEG signal ")
    plt.legend()
    plot_dir = os.path.join('E:/ccdt_data', 'snr_analysis', 'kurtosis')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, f'{NAME}.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    # 绘制PSD图
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd_mean, color='blue')
    plt.xlabel("Frequency  (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.title("Mean PSD of EEG Signal (All Channels)")

    # 限制 x 轴范围到 [0, 200] Hz
    plt.xlim(0, 200)
    plt.axvline(x=60, color='r', linestyle='--', linewidth=0.8, label="60 Hz Noise")
    plot_dir = os.path.join('E:/ccdt_data', 'snr_analysis', 'psd_0310')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, f'{NAME}.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close()

    # 显示SNR
    print(f"平均SNR: {np.mean(snr_values):.2f} dB")
    print(f"各通道SNR: {snr_values}")


# 运行主函数
if __name__ == "__main__":
    for NAME in NAMES:
        main(NAME)  # 请修改为你的文件名