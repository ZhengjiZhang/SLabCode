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

folder_path = 'E:\\ccdt_data'
data_path = 'ccdt_data'

testset = 0

max_count = 100

if testset == 1:
    NAMES = ['HUP179']
else:
    #exclude_files = {'HUP139'} #信号差的，目前还没用到

    all_files = os.listdir(os.path.join('E:/ccdt_data',data_path))
    NAMES = sorted([file for file in all_files if file.startswith('HUP')], reverse=True)
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





def main(NAME,shuffle = 0):
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
    filtered_delay = delay[(delay == 500) | (delay == 1500)]

    y = np.where(filtered_delay == 500, 0, 1)

    if shuffle == 1:
        max_dp2_distribution = []

        for single_count in range(max_count):
            start_time = time.time()  # 记录开始时间

            y = np.random.permutation(y)#shuffle labels

            ds_squared = []
            for time_bin in range(smoothed_power_3d.shape[2]):
                power_2d = smoothed_power_3d[...,time_bin]
                d_prime_squared = PLS_and_d_prime_squared(X=power_2d,y=y)
                ds_squared.append(d_prime_squared)
            max_dp2_shuffled = max(ds_squared)
            max_dp2_distribution.append(max_dp2_shuffled)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            print(f'Iteration {single_count + 1}/{max_count} completed in {elapsed_time:.2f} seconds.')
        return   max_dp2_distribution

    elif shuffle == 0:
        ds_squared = []
        for time_bin in range(smoothed_power_3d.shape[2]):
            power_2d = smoothed_power_3d[...,time_bin]
            d_prime_squared = PLS_and_d_prime_squared(X=power_2d,y=y)
            ds_squared.append(d_prime_squared)
        max_dp2 = max(ds_squared)
        print(f'Max d\'²: {max_dp2:.2f}')
        return max_dp2
    print('wrong shuffle value!')
    return None
    time_bins = np.linspace(0, 5, len(ds_squared), endpoint=True)

    plt.figure(figsize=(10, 6))
    plt.plot(time_bins, ds_squared, linestyle='-', color='black', label="d'² values")
    plt.axvline(x=1, color='gray', linestyle='--' )
    plt.axvline(x=1.5, color='gray', linestyle='--')
    plt.axvline(x=2.5, color='gray', linestyle='--')

    plt.xlabel('Time Bin')
    plt.ylabel("d'² Value")
    plt.title("d'² Value Over Time Bins")
    plt.legend()
    plt.grid(False)
    plt.show()




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

    '''# 对 PLS 投影结果进行归一化
    scaling_factors = np.linalg.norm(X_set1_pls, axis=0, keepdims=True)  # 每列的范数
    normalized_X_set1_pls = X_set1_pls / scaling_factors  # 对列归一化'''

    # 4. 使用第一个集合的均值和标准差将第二个集合进行标准化
    X_set2_scaled = scaler.transform(X_set2)

    # 5. 将第二个集合投影到第一个集合的PLS降维基础上
    X_set2_pls = pls.transform(X_set2_scaled)
    normalized_X_set2_pls = X_set2_pls #/ scaling_factors  # 使用同样的因子归一化

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

    for NAME in NAMES:


        max_dp2_distribution = main(NAME,shuffle = 1)
        max_dp2_distribution = np.array(max_dp2_distribution)
        max_dp2 = main(NAME,shuffle = 0)

        shuffled_save_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','shuffled100_0211')

        if not os.path.exists(shuffled_save_dir):
            os.makedirs(shuffled_save_dir)
        shuffle_save_path = os.path.join(shuffled_save_dir, f'{NAME}_shuffled100.npy')
        np.save(shuffle_save_path, max_dp2_distribution)
        save_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','max_dp2_0211')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{NAME}_max_dp2.npy')
        np.save(save_path, max_dp2)
        print(f'{NAME} saved successfully!')
        '''
        # 计算分布的均值和标准差
        mean_dist = np.mean(max_dp2_distribution)
        std_dist = np.std(max_dp2_distribution)

        # 计算 p 值（未打乱值大于随机分布值的比例）
        p_value = 1 - norm.cdf(max_dp2, loc=mean_dist, scale=std_dist)

        # 创建图像
        plt.figure(figsize=(10, 6))

        # 绘制随机分布的散点图
        #plt.scatter(range(len(max_dp2_distribution)), max_dp2_distribution, alpha=0.6, label='Shuffled Max Values (Scatter)', color='blue')

        # 绘制分布的直方图
        count, bins, _ = plt.hist(max_dp2_distribution, bins=30, density=False, alpha=0.6, color='skyblue', label='Shuffled Distribution (Histogram)')

        # 绘制正态分布拟合曲线
        x = np.linspace(min(bins), max(bins), 100)
        y = norm.pdf(x, loc=mean_dist, scale=std_dist)
        plt.plot(x, y, color='red', label='Gaussian Fit (Shuffled)')

        # 标注未打乱数据点
        plt.axvline(max_dp2, color='black', linestyle='--', label=f'Unshuffled Max: {max_dp2:.2f}')
        plt.text(max_dp2, max(y)/2, f'p = {p_value:.3f}', color='black', ha='right', fontsize=10)

        # 设置图表标题和标签
        plt.title("Max d'² Distribution: Shuffled vs. Unshuffled")
        plt.xlabel("d'² Values")
        plt.ylabel("Density / Counts")
        plt.legend()

        # 调整布局
        plt.tight_layout()

        # 显示图像
        plt.show()
        '''

        completion_time = datetime.datetime.now()
        print(f'Completed at: {completion_time.strftime("%Y-%m-%d %H:%M:%S")}')
