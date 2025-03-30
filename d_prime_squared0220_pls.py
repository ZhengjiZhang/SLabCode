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
from sklearn.decomposition import PCA

folder_path = 'E:\\ccdt_data'
data_path = 'ccdt_data'

testset = 0

if testset == 1:
    NAMES = ['HUP179']
elif testset == 0:
    #exclude_files = {'HUP139'} #信号差的，目前还没用到

    all_files = os.listdir(os.path.join('E:/ccdt_data',data_path))
    NAMES = [file for file in all_files if file.startswith('HUP')]
    print(NAMES)

else:
    NAMES = ['HUP133','HUP179']
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

    
    RT_s = RT[delay == 500]
    RT_l = RT[delay == 1500]

    valid_RT_s = RT_s[RT_s > 0]
    valid_RT_l = RT_l[RT_l > 0]
    FA_s = np.sum(RT_s < 0) / len(RT_s)
    FA_l = np.sum(RT_l < 0) / len(RT_l)
    delta_FA = FA_l - FA_s
    delta_RT = np.mean(valid_RT_l) - np.mean(valid_RT_s)

    # Load smoothed power data
    power_3d_path = os.path.join(folder_path, 'smoothed_power_3d')
    smoothed_power_3d = np.load(os.path.join(power_3d_path, f'smoothed_power_3d{NAME}.npy'))
    smoothed_power_3d = smoothed_power_3d[(delay == 500) | (delay == 1500),...]
    print(smoothed_power_3d.shape)



    # Define RT list and convert RTs into number of samples
    ab_rt_lst = delay + RT + 1000
    ab_num_lst = [int(ab_rt * sfreq / 1000) for ab_rt in ab_rt_lst]
    ab_rt_seconds =ab_rt_lst/1000


    # Get valid delays and initialize segment length (500ms)
    dt = int(500 * sfreq / 1000)
    filtered_delay = delay[(delay == 500) | (delay == 1500)]
    y = np.where(filtered_delay == 500, 0, 1)
    ds_squared = []
    pls_weights_all = []
    pca_weights_all = []
    for time_bin in range(smoothed_power_3d.shape[2]):
        power_2d = smoothed_power_3d[..., time_bin]

        # 分别计算 d'²
        d_prime_squared,pls_weights,pca_weights = PLS_and_d_prime_squared(X=power_2d, y=y)

        # 存储 d'²
        ds_squared.append(d_prime_squared)
        pls_weights_all.append(pls_weights)
        pca_weights_all.append(pca_weights)
    
    first_electrode_weights = []
    
    for pls_weights in pls_weights_all:
        first_electrode_weights.append(pls_weights[0][0])

    plt.figure(figsize=(10, 6))
    time_bins = np.linspace(0, 5, len(first_electrode_weights), endpoint=True)  # 根据 ds_squared 长度生成时间轴
    
    plt.plot(time_bins,first_electrode_weights, label=f'{NAME}')

    plt.xlabel('Time Bins', fontsize=20, labelpad=20)
    plt.ylabel('First Electrode Weight in PLS', fontsize=20, labelpad=20)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared', 'pls_weights_0220')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, f'first_electrode_weights_{NAME}.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f'First electrode weights in PLS {NAME} saved at {plot_save_path}')
    
    first_electrode_weights_pca = []
    
    for pca_weights in pca_weights_all:
        first_electrode_weights_pca.append(pca_weights[0][0])

    plt.figure(figsize=(10, 6))
    time_bins = np.linspace(0, 5, len(first_electrode_weights_pca), endpoint=True)  # 根据 ds_squared 长度生成时间轴
    
    plt.plot(time_bins,first_electrode_weights_pca, label=f'{NAME}')

    plt.xlabel('Time Bins', fontsize=20, labelpad=20)
    plt.ylabel('First Electrode Weight in PCA', fontsize=20, labelpad=20)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared', 'pca_weights_0220')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, f'first_electrode_weights_{NAME}.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f'First electrode weights in PLS {NAME} saved at {plot_save_path}')
    return ds_squared,pls_weights_all



def Norm_and_d_prime_squared(X, y):

    # 1. 将数据分成两个集合
    X_set1, X_set2, y_set1, y_set2 = train_test_split(X, y, test_size=0.5, random_state=42)

    # 2. 在第一个集合上进行Z-score标准化
    scaler = StandardScaler()
    X_set1_scaled = scaler.fit_transform(X_set1)



    # 4. 使用第一个集合的均值和标准差将第二个集合进行标准化
    X_set2_scaled = scaler.transform(X_set2)


    # 假设y为0和1两类，用于计算d'
    X_set2_classA = X_set2_scaled[y_set2 == 0]
    X_set2_classB = X_set2_scaled[y_set2 == 1]

    d_prime_squared = calculate_d_prime_squared(X_set2_classA, X_set2_classB)
    #print(f'd_prime_squared={d_prime_squared}')
    return d_prime_squared



def PLS_and_d_prime_squared(X, y):

    # 1. 将数据分成两个集合
    X_set1, X_set2, y_set1, y_set2 = train_test_split(X, y, test_size=0.5, random_state=42)

    # 2. 在第一个集合上进行Z-score标准化
    scaler = StandardScaler()
    X_set1_scaled = scaler.fit_transform(X_set1)

    # 3. 使用第一个集合上的标准化数据进行PLS降维
    n_components = 5  # 设置PLS成分数
    pls = PLSRegression(n_components=n_components)
    X_set1_pls = pls.fit_transform(X_set1_scaled, y_set1)[0]  # 得到降维后的数据

    pca = PCA(n_components=n_components)
    X_set1_pca = pca.fit_transform(X_set1_scaled)  # 得到降维后的数据
    pca_weights = pca.components_

   
    # 4. 使用第一个集合的均值和标准差将第二个集合进行标准化
    X_set2_scaled = scaler.transform(X_set2)

    # 5. 将第二个集合投影到第一个集合的PLS降维基础上
    X_set2_pls = pls.transform(X_set2_scaled)
    normalized_X_set2_pls = X_set2_pls  # 使用同样的因子归一化

    # 假设y为0和1两类，用于计算d'
    X_set2_classA = normalized_X_set2_pls[y_set2 == 0]
    X_set2_classB = normalized_X_set2_pls[y_set2 == 1]
    d_prime_squared = calculate_d_prime_squared(X_set2_classA, X_set2_classB)
    #print(f'd_prime_squared={d_prime_squared}')
    
    # 返回 d'² 和 PLS 权重
    pls_weights = pls.x_weights_
    return d_prime_squared,pls_weights,pca_weights



# 在这里，可以根据 X_set2_pls 的结果计算各类指标
def calculate_d_prime_squared(X_class_A, X_class_B):

    # Step 1: 计算均值向量
    mu_A = np.mean(X_class_A, axis=0)
    mu_B = np.mean(X_class_B, axis=0)
    delta_mu = mu_A - mu_B  # Δμ

    # Step 2: 计算协方差矩阵 Σ_A 和 Σ_B
    cov_A = np.cov(X_class_A, rowvar=False)
    cov_B = np.cov(X_class_B, rowvar=False)

    # 计算平均噪声协方差矩阵 Σ
    cov_avg = 0.5 * (cov_A + cov_B)

    # Step 3: 计算 Fisher 线性判别方向 w_opt
    # 使用伪逆来处理可能的奇异矩阵
    w_opt = np.linalg.pinv(cov_avg).dot(delta_mu)

    # Step 4: 计算 (d')^2
    d_prime_squared = delta_mu.T.dot(w_opt)

    #print(f"(d')^2: {d_prime_squared}")
    return d_prime_squared

if __name__ == '__main__':
    all_ds_squared = []
    pls_weights_all_names = []
    for NAME in NAMES:
        ds_squared,pls_weights_all = main(NAME)

        all_ds_squared.append(ds_squared)
        pls_weights_all_names.append(pls_weights_all)




    completion_time = datetime.datetime.now()
    print(f'Completed at: {completion_time.strftime("%Y-%m-%d %H:%M:%S")}')