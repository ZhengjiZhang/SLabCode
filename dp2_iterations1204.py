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
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import colors

folder_path = 'E:\\ccdt_data'
data_path = 'ccdt_data'

testset = 1

if testset == 1:
    NAMES = ['HUP179']
else:
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
    rt = beh_df['RT'].values
    
    # Define RT list and convert RTs into number of samples
    ab_rt_lst = beh_df['delay'] + beh_df['RT'] + 1000
    ab_num_lst = [(ab_rt * sfreq / 1000) for ab_rt in ab_rt_lst]

    # Load smoothed power data
    power_3d_path = os.path.join(folder_path, 'smoothed_power_3d')
    smoothed_power_3d = np.load(os.path.join(power_3d_path, f'smoothed_power_3d{NAME}.npy'))
    smoothed_power_3d = smoothed_power_3d[(delay == 500) | (delay == 1500),...]

    print("Original shape:", smoothed_power_3d.shape)
    
    # 对时间维度进行循环平移 (circular shift)
    for i in range(smoothed_power_3d.shape[0]):  # 遍历第一个维度
        for j in range(smoothed_power_3d.shape[1]):  # 遍历第二个维度
            shift_amount = np.random.randint(0, smoothed_power_3d.shape[2])  # 生成随机平移量
            smoothed_power_3d[i, j, :] = np.roll(smoothed_power_3d[i, j, :], shift_amount)  # 循环平移

    print("Shuffled shape:", smoothed_power_3d.shape)

    # Get valid delays and initialize segment length (500ms)
    filtered_delay = delay[(delay == 500) | (delay == 1500)]
    y = np.where(filtered_delay == 500, 0, 1)

    y = np.random.permutation(y)#shuffle labels


    ds_squared = []
    for time_bin in range(smoothed_power_3d.shape[2]):
        power_2d = smoothed_power_3d[...,time_bin]
        d_prime_squared = PLS_and_d_prime_squared(X=power_2d,y=y)
        ds_squared.append(d_prime_squared)
    max_index=np.argmax(ds_squared)
    max_value = ds_squared[max_index]
    print(f'max_value={max_value},max_index={int(max_index/sfreq * 1000)}ms')

    dt = int(250 * sfreq / 1000)

    smoothed_power_cutted_mean = smoothed_power_3d[...,max_index-dt:max_index+dt].mean(axis=2)   # 平均当前时间段的数据
    electrode_number = smoothed_power_3d.shape[1]
    electrode_indices = range(electrode_number)
    sampled_permutations = []  # 用于存储所有的随机排列
    num_permutations = 100  # 生成 100
    for _ in range(num_permutations):  # 只生成 1000 个随机排列
        sampled_permutations.append(tuple(np.random.permutation(electrode_indices)))
    print(len(sampled_permutations))

    dp2s_all = []  # 用于存储所有的 d'²
    for j,perm in enumerate(sampled_permutations):

      
        smoothed_power_2d_rearranged = smoothed_power_cutted_mean[:, list(perm)]  # 包含前i+1个电极的数据
        dp2s_bin = []

        # 对每个电极数进行d'²计算
        for i in range(electrode_number):
            smoothed_power_cutted = smoothed_power_2d_rearranged[:, :i + 1]  # 包含前i+1个电极的数据
            if smoothed_power_cutted.ndim == 1:
                smoothed_power_cutted = smoothed_power_cutted[:, np.newaxis]  # 将1D数组调整为2D

            # 计算d'²
            d_prime_squared = Norm_and_d_prime_squared(X=smoothed_power_cutted, y=y)
            dp2s_bin.append(d_prime_squared)
        dp2s_all.append(dp2s_bin)
        if j % int(0.1*num_permutations) == 0:
            print(f'the {j}th cycle is done!')


    # 计算均值和标准误差
    mean_dp2s = np.mean(dp2s_all, axis=0)
    std_dp2s = np.std(dp2s_all, axis=0)
    ci_95 = 1.96 * std_dp2s / np.sqrt(num_permutations)  # 95% 置信区间

    plt.figure(figsize=(10, 6))
    plt.plot(mean_dp2s, color='blue', label='Mean d\'² Value')
    plt.fill_between(range(electrode_number), mean_dp2s - ci_95, mean_dp2s + ci_95, color='blue', alpha=0.3, label='95% CI')


    # 添加标题和标签
    plt.xlabel('Electrode Nums', fontsize=20, labelpad=20)
    plt.ylabel("d'² Value", fontsize=20, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared', 'iterations_sorted_1205_2')
    os.makedirs(plot_dir, exist_ok=True)
    plot_save_path = os.path.join(plot_dir, f'{NAME}_dp2_permutations_mean.png')
    plt.savefig(plot_save_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, d2s_bin in enumerate(dp2s_all):
        plt.plot(d2s_bin, color = 'blue', alpha=0.7)  # 绘制每一组数据

    # 添加标题和标签
    plt.xlabel('Electrode Nums', fontsize=20, labelpad=20)
    plt.ylabel("d'² Value", fontsize=20, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared', 'iterations_sorted_0126')
    os.makedirs(plot_dir, exist_ok=True)
    plot_save_path = os.path.join(plot_dir, f'{NAME}_dp2_permutations.png')
    plt.savefig(plot_save_path)
    plt.close()

    print(f' d_prime_squared curves saved at {plot_save_path}')





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


def d2_1d(X_class_A, X_class_B):
    if X_class_A.ndim !=1 or X_class_B.ndim !=1:
        print("Warning! the dim for 1D analysis is not 1!")
    # Step 1: 计算均值向量
    mu_A = np.mean(X_class_A, axis=0)
    mu_B = np.mean(X_class_B, axis=0)
    delta_mu = mu_A - mu_B  # Δμ

    var_A = np.var(X_class_A, axis=0)
    var_B = np.var(X_class_B, axis=0)
    var_avg = 0.5 * (var_A + var_B)

    d_prime_squared = delta_mu**2/var_avg
    return d_prime_squared

# 在这里，可以根据 X_set2_pls 的结果计算各类指标
def calculate_d_prime_squared(X_class_A, X_class_B):

    # 确保输入数据为二维数组
    if X_class_A.ndim == 1:
        X_class_A = X_class_A[:, np.newaxis]
    if X_class_B.ndim == 1:
        X_class_B = X_class_B[:, np.newaxis]
    # Step 1: 计算均值向量
    mu_A = np.mean(X_class_A, axis=0)
    mu_B = np.mean(X_class_B, axis=0)
    delta_mu = mu_A - mu_B  # Δμ

    # Step 2: 计算协方差矩阵 Σ_A 和 Σ_B
    cov_A = np.cov(X_class_A, rowvar=False)
    cov_B = np.cov(X_class_B, rowvar=False)

    # 检查协方差矩阵的维度，确保其为 2D
    if cov_A.ndim == 0:
        cov_A = np.array([[cov_A]])
    if cov_B.ndim == 0:
        cov_B = np.array([[cov_B]])

    # 计算平均噪声协方差矩阵 Σ
    cov_avg = 0.5 * (cov_A + cov_B) + 1e-6 * np.eye(cov_A.shape[0])

    cov_avg = np.nan_to_num(cov_avg)

    # Step 3: 计算 Fisher 线性判别方向 w_opt
    # 使用伪逆来处理可能的奇异矩阵
    w_opt = np.linalg.pinv(cov_avg).dot(delta_mu)

    # Step 4: 计算 (d')^2
    d_prime_squared = delta_mu.T.dot(w_opt)

    #print(f"(d')^2: {d_prime_squared}")
    return d_prime_squared

if __name__ == '__main__':
    all_ds_squared = []  # 用于存储所有的 ds_squared

    for NAME in NAMES:
        main(NAME)
        '''
        all_ds_squared.append(ds_squared)
        plt.figure(figsize=(10, 6))
        time_bins = np.linspace(0, 5, len(ds_squared), endpoint=True)  # 根据 ds_squared 长度生成时间轴

        plt.plot(time_bins, ds_squared, linestyle='-', color='orange')  # 每个 ds_squared 单独绘制

        # 设置图例和轴标签
        plt.axvline(x=1, color='gray', linestyle='--' )
        plt.axvline(x=1.5, color='gray', linestyle='--')
        plt.axvline(x=2.5, color='gray', linestyle='--')
        plt.xlim(0,5)
        #plt.ylim(0,16)

        plt.xlabel("Time (s)")
        plt.ylabel("d'² values")
        plt.title("d'² values over time bins ")
        plt.legend()

        plt.tight_layout()
        # Save the figure
        plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','reduced10_1114')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_save_path = os.path.join(plot_dir, f'{NAME}_d_prime_squared.png')



        # Save the plot
        plt.savefig(plot_save_path)


        print(f' d_prime_squared curves saved at {plot_save_path}')
        #plt.show()
        plt.close()'''



    '''
    plt.figure(figsize=(10, 6))
    for i, ds_squared in enumerate(all_ds_squared):
        time_bins = np.linspace(0, 5, len(ds_squared), endpoint=True)  # 根据 ds_squared 长度生成时间轴
        alpha_value = 1.0 - (i / len(all_ds_squared))  # 每条曲线的透明度逐渐减小

        plt.plot(time_bins, ds_squared, linestyle='-', color='b',alpha=alpha_value)  # 每个 ds_squared 单独绘制

    # 设置图例和轴标签
    plt.axvline(x=1, color='gray', linestyle='--' )
    plt.axvline(x=1.5, color='gray', linestyle='--')
    plt.axvline(x=2.5, color='gray', linestyle='--')
    plt.xlim(0,5)
    #plt.ylim(0,16)

    plt.xlabel("Time (s)")
    plt.ylabel("d'² values")
    plt.title("d'² values over time bins ")
    plt.legend()

    plt.tight_layout()
    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_save_path = os.path.join(plot_dir, 'd_prime_squared.png')



    # Save the plot
    plt.savefig(plot_save_path)


    print(f' d_prime_squared curves saved at {plot_save_path}')
    plt.show()
    plt.close()
    '''