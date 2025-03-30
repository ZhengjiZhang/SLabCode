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
    # 将 trials 分为前后两半
    num_trials = smoothed_power_3d.shape[0]
    half_trials = num_trials // 2

    # 前半部分 trials
    smoothed_power_3d_first_half = smoothed_power_3d[:half_trials, :, :]
    # 后半部分 trials
    smoothed_power_3d_second_half = smoothed_power_3d[half_trials:, :, :]

    # Define RT list and convert RTs into number of samples
    ab_rt_lst = delay + RT + 1000
    ab_num_lst = [int(ab_rt * sfreq / 1000) for ab_rt in ab_rt_lst]
    ab_rt_seconds =ab_rt_lst/1000


    # Get valid delays and initialize segment length (500ms)
    dt = int(500 * sfreq / 1000)
    filtered_delay = delay[(delay == 500) | (delay == 1500)]
    y = np.where(filtered_delay == 500, 0, 1)
    ds_squared_first_half = []
    ds_squared_second_half = []
    for time_bin in range(smoothed_power_3d.shape[2]):
        power_2d_first = smoothed_power_3d_first_half[..., time_bin]
        power_2d_second = smoothed_power_3d_second_half[..., time_bin]

        # 分别计算 d'²
        d_prime_squared_first = PLS_and_d_prime_squared(X=power_2d_first, y=y[:half_trials])
        d_prime_squared_second = PLS_and_d_prime_squared(X=power_2d_second, y=y[half_trials:])

        # 存储 d'²
        ds_squared_first_half.append(d_prime_squared_first)
        ds_squared_second_half.append(d_prime_squared_second)
    
    ds_squared_diff = [second - first for first, second in zip(ds_squared_first_half, ds_squared_second_half)]
    print(f"{NAME} d'² calculated for both halves and differences computed!")
    bin_distance = 0.05
    time_bins = np.linspace(0, 5, len(ds_squared_diff), endpoint=True)
    # Create histogram for reaction times


    '''
    # 双轴绘图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴：绘制 d'² 差值的折线图
    color1 = 'orange'
    ax1.plot(time_bins, ds_squared_diff, color=color1, label="d'² diff", linestyle='-')
    ax1.axvline(x=1, color='gray', linestyle='--')
    ax1.text(1, ax1.get_ylim()[1], 'S1', color='black', ha='center', va='bottom', fontsize=20)

    ax1.axvline(x=1.5, color='gray', linestyle='--')
    ax1.text(1.5, ax1.get_ylim()[1], 'S2', color='black', ha='center', va='bottom', fontsize=20)

    ax1.axvline(x=2.5, color='gray', linestyle='--')
    ax1.text(2.5, ax1.get_ylim()[1], "S2'", color='black', ha='center', va='bottom', fontsize=20)
    ax1.set_xlabel('Time (s)',fontsize=20,labelpad=20)
    ax1.set_ylabel("d'² Difference",fontsize=20,labelpad=20)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    ax1.set_xlim(0, 5)  #  x 轴限制

    plt.tight_layout()

    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','diff_dim5_RT_1204')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, f'{NAME}_dp2_diff.png')



    # Save the plot
    plt.savefig(plot_save_path,dpi = 300)


    print(f' d_prime_squared curves saved at {plot_save_path}')
    plt.close()'''
    return ds_squared_diff, delta_FA, delta_RT



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
    d_prime_squared = calculate_d_prime_squared(X_set2_classA, X_set2_classB)
    #print(f'd_prime_squared={d_prime_squared}')
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
    return d_prime_squared

if __name__ == '__main__':
    all_ds_squared_diff = []
    all_delta_FA = []
    all_delta_RT = []
    for NAME in NAMES:
        ds_squared_diff,delta_FA,delta_RT = main(NAME)
        max_val = max(ds_squared_diff)
        min_val = min(ds_squared_diff)
        range_val = max_val - min_val
        ds_squared_diff /= range_val
        all_ds_squared_diff.append(ds_squared_diff)
        all_delta_FA.append(delta_FA)
        all_delta_RT.append(delta_RT)


    time_length = 5  # 假设时间总长为 5 秒

    target_points = min(len(arr) for arr in all_ds_squared_diff)  # 取最小频率为目标
    print(target_points)
    # 对每个数组进行重采样
    resampled_arrays = []
    for arr in all_ds_squared_diff:
        # 原始时间轴
        original_time = np.linspace(0, time_length, len(arr))
        # 目标时间轴
        target_time = np.linspace(0, time_length, target_points)
        # 插值到目标频率
        interpolated = interp1d(original_time, arr, kind='linear')(target_time)
        resampled_arrays.append(interpolated)


    resampled_arrays = np.array(resampled_arrays)

    all_delta_FA = np.array(all_delta_FA)
    all_delta_RT = np.array(all_delta_RT)

    time_bins = np.linspace(0, 5, len(resampled_arrays[0]), endpoint=True)  # 根据 ds_squared 长度生成时间轴

    # 根据 从大到小排序
    sorted_indices_FA = np.argsort(all_delta_FA)
    sorted_ds_squared_diff_FA = resampled_arrays[sorted_indices_FA]
    
    # 绘制每条曲线，并空出一些距离
    plt.figure(figsize=(10, 6))
    offset = 0
    for i, ds_squared_diff in enumerate(sorted_ds_squared_diff_FA):
        plt.plot(time_bins,ds_squared_diff + offset,color='orange')
        offset += 1  # 每条曲线空出一些距离

    # 添加标题和标签
    plt.xlabel("Time (s)",fontsize=20, labelpad=20)
    plt.ylabel("Normalized d'² Diff",fontsize=20,labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.yticks([]) #Δ
    plt.xlim(0,5)

    plt.axvline(x=1, color='gray', linestyle='--')
    plt.text(1, plt.ylim()[1], 'S1', color='black', ha='center', va='bottom', fontsize=20)

    plt.axvline(x=1.5, color='gray', linestyle='--')
    plt.text(1.5, plt.ylim()[1], 'S2', color='black', ha='center', va='bottom', fontsize=20)

    plt.axvline(x=2.5, color='gray', linestyle='--')
    plt.text(2.5, plt.ylim()[1], "S2'", color='black', ha='center', va='bottom', fontsize=20)

    plt.annotate('ΔFA', xy=(1.02, 1), xycoords='axes fraction', xytext=(1.02,0),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             fontsize=16, ha='center', va='center', rotation=0)
    plt.tight_layout()  # 调整布局避免重叠

    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','normalized')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, f'norm_dp2_diff_sortedFA.png')



    # Save the plot
    plt.savefig(plot_save_path,dpi = 300)

    plt.close()

    # 根据 delta_RT 从大到小排序
    sorted_indices_RT = np.argsort(all_delta_RT)
    sorted_ds_squared_diff_RT = resampled_arrays[sorted_indices_RT]

    # 绘制每条曲线，并空出一些距离
    plt.figure(figsize=(10, 6))
    offset = 0
    for i, ds_squared_diff in enumerate(sorted_ds_squared_diff_RT):
        plt.plot(time_bins,ds_squared_diff + offset,color='orange')
        offset += 1  # 每条曲线空出一些距离

    # 添加标题和标签
    plt.xlabel("Time (s)", fontsize=20, labelpad=20)
    plt.ylabel("Normalized d'² Diff", fontsize=20, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.yticks([])  # 隐藏 y 轴刻度
    plt.xlim(0,5)

    plt.axvline(x=1, color='gray', linestyle='--')
    plt.text(1, plt.ylim()[1], 'S1', color='black', ha='center', va='bottom', fontsize=20)

    plt.axvline(x=1.5, color='gray', linestyle='--')
    plt.text(1.5, plt.ylim()[1], 'S2', color='black', ha='center', va='bottom', fontsize=20)

    plt.axvline(x=2.5, color='gray', linestyle='--')
    plt.text(2.5, plt.ylim()[1], "S2'", color='black', ha='center', va='bottom', fontsize=20)
    
    plt.annotate('ΔRT', xy=(1.02, 1), xycoords='axes fraction', xytext=(1.02,0),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             fontsize=16, ha='center', va='center', rotation=0)
    plt.tight_layout()  # 调整布局避免重叠

    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared', 'normalized')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, f'norm_dp2_diff_sortedRT.png')

    # Save the plot
    plt.savefig(plot_save_path)
    plt.close()

    print(f'Normalized d\'² difference curves sorted by RT saved at {plot_save_path}')    
    completion_time = datetime.datetime.now()
    print(f'Completed at: {completion_time.strftime("%Y-%m-%d %H:%M:%S")}')