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





def main(NAME,PLS_dim):
    """
    Main function to process data, train models, and evaluate performance
    for each electrode.
    """
    print(NAME)
    
    # Load subject data
    eeg_data, eeg_metadata, beh_df, _, _ = loadSubjData('E:/ccdt_data/ccdt_data/', NAME)

    delay = beh_df['delay'].values
    sfreq = int(eeg_metadata['samplerate'])
    rt = beh_df['RT'].values
    
    # Define RT list and convert RTs into number of samples
    ab_rt_lst = beh_df['delay'] + beh_df['RT'] + 1000
    ab_num_lst = [(ab_rt * sfreq / 1000) for ab_rt in ab_rt_lst]

    # Load smoothed power data
    power_3d_path = os.path.join(folder_path, 'smoothed_power_3d')
    smoothed_power_3d = np.load(os.path.join(power_3d_path, f'smoothed_power_3d{NAME}.npy'))
    smoothed_power_3d = smoothed_power_3d[(delay == 500) | (delay == 1500),...]
    print("Original shape:", smoothed_power_3d.shape)
    '''
    # 对时间维度进行循环平移 (circular shift)
    for i in range(smoothed_power_3d.shape[0]):  # 遍历第一个维度
        for j in range(smoothed_power_3d.shape[1]):  # 遍历第二个维度
            shift_amount = np.random.randint(0, smoothed_power_3d.shape[2])  # 生成随机平移量
            smoothed_power_3d[i, j, :] = np.roll(smoothed_power_3d[i, j, :], shift_amount)  # 循环平移
    
    print("Shuffled shape:", smoothed_power_3d.shape)
    '''
    #print(smoothed_power_3d.shape)
    #time.sleep(100)
    # Get valid delays and initialize segment length (500ms)
    dt = int(500 * sfreq / 1000)
    filtered_delay = delay[(delay == 500) | (delay == 1500)]
    y = np.where(filtered_delay == 500, 0, 1)


    dp2s_set2 = []
    dp2s_set3 = []
    for time_bin in range(smoothed_power_3d.shape[2]):
        power_2d = smoothed_power_3d[...,time_bin]
        d_prime_squared_set2,d_prime_squared_set3 = PLS_and_d_prime_squared(X=power_2d,y=y,PLS_dim=PLS_dim)
        dp2s_set2.append(d_prime_squared_set2)
        dp2s_set3.append(d_prime_squared_set3)
    dp2_set2_max = max(dp2s_set2)
    dp2_set3_max = max(dp2s_set3)
    print(f"{NAME} d'2 calculated!")
    return   dp2_set2_max,dp2_set3_max
def PLS_and_d_prime_squared(X, y,PLS_dim):

    # 1. 将数据分成三个集合
    X_set1, X_remaining, y_set1, y_remaining = train_test_split(X, y, test_size=2/3, random_state=42)
    X_set2, X_set3, y_set2, y_set3 = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)


    # 2. 在第一个集合上进行Z-score标准化
    scaler = StandardScaler()
    X_set1_scaled = scaler.fit_transform(X_set1)

    # 3. 使用第一个集合上的标准化数据进行PLS降维
    n_components = PLS_dim  # 设置PLS成分数
    pls = PLSRegression(n_components=n_components)
    X_set1_pls = pls.fit_transform(X_set1_scaled, y_set1)[0]  # 得到降维后的数据

    # 对 PLS 投影结果进行归一化
    scaling_factors = np.linalg.norm(X_set1_pls, axis=0, keepdims=True)  # 每列的范数
    normalized_X_set1_pls = X_set1_pls / scaling_factors  # 对列归一化

    # 4. 使用第一个集合的均值和标准差将第二个集合进行标准化
    X_set2_scaled = scaler.transform(X_set2)

    # 5. 将第二个集合投影到第一个集合的PLS降维基础上
    X_set2_pls = pls.transform(X_set2_scaled)
    normalized_X_set2_pls = X_set2_pls / scaling_factors  # 使用同样的因子归一化

    # 假设y为0和1两类，用于计算d'
    X_set2_classA = normalized_X_set2_pls[y_set2 == 0]
    X_set2_classB = normalized_X_set2_pls[y_set2 == 1]
    w_opt,d_prime_squared_set2 = calculate_d_prime_squared(X_set2_classA, X_set2_classB)
    #print(f'd_prime_squared={d_prime_squared}')

    #使用第一个集合的均值和标准差将第三个集合进行标准化
    X_set3_scaled = scaler.transform(X_set3)
    X_set3_pls = pls.transform(X_set3_scaled)
    normalized_X_set3_pls = X_set3_pls / scaling_factors  # 使用同样的因子归一化
    X_set3_classA = normalized_X_set3_pls[y_set3 == 0]
    X_set3_classB = normalized_X_set3_pls[y_set3 == 1]
    d_prime_squared_set3 = calculate_d_prime_squared_set3(X_set3_classA, X_set3_classB,w_opt)
    return d_prime_squared_set2,d_prime_squared_set3


def calculate_d_prime_squared_set3(X_class_A, X_class_B,w_opt):
    # Step 1: 计算均值向量
    mu_A = np.mean(X_class_A, axis=0)
    mu_B = np.mean(X_class_B, axis=0)
    delta_mu = mu_A - mu_B  # Δμ
    d_prime_squared = delta_mu.T.dot(w_opt)
    return d_prime_squared


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
    return w_opt,d_prime_squared

if __name__ == '__main__':
    PLS_dims = range(2,21)

    for NAME in NAMES:
        dp2s_set2_max = []
        dp2s_set3_max = []
        for PLS_dim in PLS_dims:
            print(f'PLS_dim={PLS_dim}')
            dp2_set2_max,dp2_set3_max = main(NAME,PLS_dim)
            dp2s_set2_max.append(dp2_set2_max)
            dp2s_set3_max.append(dp2_set3_max)
        plt.figure(figsize=(10, 6))

        plt.plot(PLS_dims,dp2s_set2_max, linestyle='-', color='orange')  # 每个 ds_squared 单独绘制
        plt.plot(PLS_dims,dp2s_set3_max, linestyle='-', color='blue')  # 每个 ds_squared 单独绘制


        plt.xlabel("PLS dims", fontsize=20, labelpad=20)
        plt.ylabel("d'² values", fontsize=20, labelpad=20)
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.xticks(range(min(PLS_dims), max(PLS_dims) + 1))

        plt.tight_layout()
        # Save the figure
        plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','PLS_dims_0211')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_save_path = os.path.join(plot_dir, f'{NAME}_PLS_dims.png')



        # Save the plot
        plt.savefig(plot_save_path,dpi = 300)


        print(f' saved at {plot_save_path}')
        #plt.show()
        plt.close()

        print(f'{NAME} saved successfully!')