import pickle
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_morlet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import pandas as pd

sigma = 75

'''
data_dir = 'E:/ccdt_data/ccdt_data'
all_files = os.listdir(data_dir)
NAMES = [file for file in all_files if file.startswith('HUP')]
print(NAMES)'''
NAMES = ['HUP191']

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)
    return obj

def loadSubjData(save_dir,subj):#,

    """ Inputs
    subj ... ID of the subject eg 'HUP179' 
    save_dir ... directory where the data are stored 
    Ashwin G. Ramayya 04/03/2023
    """

    
    # look for save file
    if (os.path.exists(save_dir+subj)==True):#
        #load file if it exists
        thisSubjData = (load_pickle(save_dir+subj))#save_dir+

        # unpack the list to get the files
        eeg_data = thisSubjData[0]
        eeg_metadata = thisSubjData[1]
        beh_df = thisSubjData[2]
        anatDThisSubj = thisSubjData[3]
        anatDfThisSubj_yeo  = thisSubjData[4]
        
        return eeg_data,eeg_metadata,beh_df,anatDThisSubj,anatDfThisSubj_yeo        
    else:
        print('Error: Data for this subject not found')

def plot_data(data, sfreq, electrode_index = 20):
    time = np.linspace(0, len(data[electrode_index-1]) / sfreq, num=len(data[electrode_index-1]))
    plt.figure(figsize=(10, 4))
    plt.plot(time, data[electrode_index-1], label=f'Electrode {electrode_index}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    #plt.title('Smoothed EEG Data')
    plt.legend()
    plt.show()

def main(NAME):
    eeg_metadata = loadSubjData('E:/ccdt_data/ccdt_data/',NAME)[1]
   

    sfreq = int(eeg_metadata['samplerate'])
    print(sfreq)

    eeg_data = loadSubjData('E:/ccdt_data/ccdt_data/',NAME)[0]


    #trialxelecxtime转换后结果
    eeg_data = eeg_data.transpose(0,2,1)

    # 定义频率范围和小波的周期数
    freqs = np.logspace(np.log10(70), np.log10(200), 5)
    n_cycles = freqs / 3  # 每个频率的周期数，可以根据需要调整

    tfr = mne.time_frequency.tfr_array_morlet(eeg_data, sfreq=sfreq,
                                              freqs=freqs,n_cycles=n_cycles,
                                              output='complex')



    wave_complex = np.moveaxis(tfr,(2,3,0),(0,1,2)).squeeze()

    wave_power = np.abs(wave_complex) ** 2
    wave_power = np.log10(wave_power)


    #这里是重新计算powermean的方法
    # 初始化一个与其他维度相同形状的数组用于存储均值
    power_mean = np.zeros((wave_power.shape[1], wave_power.shape[2], wave_power.shape[3]))
    # 对每个频率单独计算均值，并累加到 power_mean
    for i in range(wave_power.shape[0]):
        power_mean += np.nan_to_num(wave_power[i], nan=0.0)

    # 计算平均值
    power_mean /= wave_power.shape[0]

    # 如果需要，可以进一步将 NaN 值重新引入
    power_mean[np.isnan(power_mean)] = np.nan


    power_mean = np.transpose(power_mean, (1, 2, 0))#trialxelecxtime

    folder_path = 'E:/ccdt_data' 
    power_path = os.path.join(folder_path,f'power_mean')
     
    if not os.path.exists(power_path):
        os.makedirs(power_path)

    np.save(os.path.join(power_path, f'power_mean{NAME}.npy'),power_mean)
    print(f'{NAME}power_mean saved!')

    smoothed_power_3d = np.zeros_like(power_mean)


    timelength = 500 * sfreq / 1000
    dt = int(500 * sfreq / 1000)



    for i in range(power_mean.shape[0]):
        smoothed_power_3d[i] += gaussian_filter1d(power_mean[i], sigma=sigma, axis=1)  # 假设 wavePower 是按 (electrode, time) 组织的
    power_3d_path = os.path.join(folder_path,f'smoothed_power_3d')
     
    if not os.path.exists(power_3d_path):
        os.makedirs(power_3d_path)

    np.save(os.path.join(power_3d_path, f'smoothed_power_3d{NAME}.npy'),smoothed_power_3d)
    print(f'{NAME}power_3d saved!')
if __name__ == '__main__':
    for NAME in NAMES:
        main(NAME)