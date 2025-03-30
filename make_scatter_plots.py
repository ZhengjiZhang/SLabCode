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

folder_path = 'E:\\ccdt_data'
data_path = 'ccdt_data'

testset = 0


if testset == 1:
    NAMES = ['HUP179']
else:
    #exclude_files = {'HUP139'} #信号差的，目前还没用到

    all_files = os.listdir(os.path.join('E:/ccdt_data',data_path))
    NAMES = [file for file in all_files if file.startswith('HUP')]
    print(NAMES)

electrode_index_specific = 10000#large number means useless now

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)
    return obj

def loadSubjData(save_dir, subj):
    """加载特定受试者的数据"""
    if os.path.exists(save_dir + subj):
        thisSubjData = load_pickle(save_dir + subj)
        eeg_data = thisSubjData[0]
        eeg_metadata = thisSubjData[1]
        beh_df = thisSubjData[2]
        anatDThisSubj = thisSubjData[3]
        anatDfThisSubj_yeo = thisSubjData[4]
        return eeg_data, eeg_metadata, beh_df, anatDThisSubj, anatDfThisSubj_yeo
    else:
        print('Error: Data for this subject not found')





def main(NAME):
    """
    Main function to process data, train models, and evaluate performance
    for each electrode.
    """
    print(NAME)

    # Load subject data
    eeg_data, eeg_metadata, beh_df, _, _ = loadSubjData('E:/ccdt_data/ccdt_data/', NAME)

    delay = beh_df['delay'].values
    sfreq = int(eeg_metadata['samplerate'])
    RT = beh_df['RT'].values


    # Load smoothed power data
    power_3d_path = os.path.join(folder_path, 'smoothed_power_3d')
    smoothed_power_3d = np.load(os.path.join(power_3d_path, f'smoothed_power_3d{NAME}.npy'))
    smoothed_power_3d = smoothed_power_3d[(delay == 500) | (delay == 1500),...]
    print(smoothed_power_3d.shape)



    # Define RT list and convert RTs into number of samples
    ab_rt_lst = delay + RT + 1000



    RT_s = RT[delay == 500]
    RT_l = RT[delay == 1500]

    valid_RT_s = RT_s[RT_s > 0]
    valid_RT_l = RT_l[RT_l > 0]
    FA_s = np.sum(RT_s < 0) / len(RT_s)
    FA_l = np.sum(RT_l < 0) / len(RT_l)
    delta_FA = FA_l - FA_s
    delta_RT = np.mean(valid_RT_l) - np.mean(valid_RT_s)
    print(np.mean(valid_RT_l), np.mean(valid_RT_s))
    return delta_FA, delta_RT


if __name__ == '__main__':
    special_names_1 = ['HUP139','HUP154']
    special_names_2 = ['HUP157','HUP160','HUP168','HUP178']
    delta_FA_list = []
    delta_RT_list = []
    colors = []
    for NAME in NAMES:
        delta_FA, delta_RT = main(NAME)
        delta_FA_list.append(delta_FA)
        delta_RT_list.append(delta_RT)
        if NAME in special_names_1:
            colors.append('red')  # 特定的 HUP 数据点标红
        elif NAME in special_names_2:
            colors.append('blue')
        else:
            colors.append('gray')  # 其他数据点为灰色
    fig, ax = plt.subplots()

    # 绘制每个点，根据标签设置颜色和样式
    ax.scatter(delta_RT_list, delta_FA_list,color=colors)
    plt.xlabel('Δ RT (ms)',fontsize=20,labelpad=20)
    plt.ylabel('Δ FA rate',fontsize=20,labelpad=20)

    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.grid(False)
    plt.tight_layout()
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','scatter_plots')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, f'scatter_RT_0220_2.png')
    plt.savefig(plot_save_path,dpi = 300)