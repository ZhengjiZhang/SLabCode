import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import math
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from matplotlib import colormaps

folder_path = 'E:\\ccdt_data'
data_path = 'ccdt_data'

testset = 1

if testset == 1:
    NAMES = ['HUP146']#！！！现在不是179，记得改回来
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

    #print(smoothed_power_3d.shape)

    # Get valid delays and initialize segment length (500ms)
    dt = int(100 * sfreq / 1000)
    filtered_delay = delay[(delay == 500) | (delay == 1500)]
    y = np.where(filtered_delay == 500, 0, 1)
    d2s_all = []
    electrode_number = smoothed_power_3d.shape[1]
    #print(electrode_number)
    #time.sleep(100)
    #shuffled_indices = np.random.permutation(electrode_number)
    #shuffle_power_3d = smoothed_power_3d[:,shuffled_indices,int(500 * sfreq / 1000):int(3500 * sfreq / 1000)]
    left_point = int(1500  * sfreq / 1000)
    right_point = int(3500 * sfreq / 1000)
    timeperiod_power_3d = smoothed_power_3d[...,left_point:right_point]

    d2s_1d_all = []
    for start_time in range(0, timeperiod_power_3d.shape[2], dt):
        d2s_1d = []
        # 获取当前时间段内的平均数据
        end_time = min(start_time + dt, smoothed_power_3d.shape[2])  # 确保不会超过时间长度
        power_2d_mean = timeperiod_power_3d[..., start_time:end_time].mean(axis=2)  # 平均当前时间段的数据

        # 对每个电极数进行d'²计算
        for i in range(electrode_number):
            cutted_power_1d = power_2d_mean[:,i]  # 第i个电极的数据
            mean = np.mean(cutted_power_1d)
            std = np.std(cutted_power_1d)
            cutted_power_1d_scaled = (cutted_power_1d - mean) / std
            X_set2_classA = cutted_power_1d_scaled[y == 0]
            X_set2_classB = cutted_power_1d_scaled[y == 1]
            # 计算d'²
            d_prime_squared = d2_1d(X_class_A=X_set2_classA,X_class_B=X_set2_classB)
            d2s_1d.append(d_prime_squared)
        d2s_1d_all.append(d2s_1d)
    d2s_mean = np.mean(d2s_1d_all,axis=0)
    sorted_indices = np.argsort(d2s_mean)[::-1]  # 降序排序，[::-1] 反转
    sorted_power_3d = smoothed_power_3d[:,sorted_indices,left_point:right_point]


    for start_time in range(0, sorted_power_3d.shape[2], dt):
        d2s_bin = []
        # 获取当前时间段内的平均数据
        end_time = min(start_time + dt, smoothed_power_3d.shape[2])  # 确保不会超过时间长度
        power_2d_mean = sorted_power_3d[..., start_time:end_time].mean(axis=2)  # 平均当前时间段的数据

        # 对每个电极数进行d'²计算
        for i in range(electrode_number):
            cutted_power_2d = power_2d_mean[:, :i + 1]  # 包含前i+1个电极的数据
            if cutted_power_2d.ndim == 1:
                cutted_power_2d = cutted_power_2d[:, np.newaxis]  # 将1D数组调整为2D

            # 计算d'²
            d_prime_squared = Norm_and_d_prime_squared(X=cutted_power_2d, y=y)
            d2s_bin.append(d_prime_squared)

        # 对 d'² 的结果进行归一化
        min_val = d2s_bin[0]
        max_val = d2s_bin[-1]
        d2s_bin_normalized = [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in d2s_bin]
        # 存储归一化结果
        d2s_all.append(d2s_bin_normalized)

        # 打印进度
        print(f'Time window starting at {int(start_time * 1000 /sfreq )} is done!')



    cmap = plt.colormaps.get_cmap('rainbow')  # 使用彩虹色渐变
    norm = Normalize(vmin=0, vmax=int(len(d2s_all)*dt/sfreq))  # 根据时间窗口数量设置归一化范围
    fig, ax = plt.subplots(figsize=(10, 6))


    # 对 d2s_all 中的每个 d2s_bin 作图
    for i, d2s_bin in enumerate(d2s_all):
        if i < len(d2s_all) / 2:  # 只展示前一半的曲线

            # 使用透明度渐变，随着时间bin递增透明度加深
            color = cmap(norm(i*dt/sfreq))  # 根据 i 生成颜色

            ax.plot(d2s_bin, color=color, alpha=0.8)  # 设置曲线颜色和透明度

    # 设置图表标题和标签
    ax.set_xlabel("Electrode Numbers",fontsize=16,labelpad=10)
    ax.set_ylabel("Normalized d'²",fontsize=16,labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=14)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 不需要具体的数组数据
    cbar = plt.colorbar(sm,ax = ax, orientation="horizontal", fraction=0.05, pad=0.2)
    cbar.set_label("Time Bins",fontsize=16,labelpad=10)

    cbar.ax.axvline(x=0.0, color='gray', linestyle='--')
    cbar.ax.text(0.0, 1.1, 'S2', color='black', ha='center', va='bottom', fontsize=12, transform=cbar.ax.transAxes)

    cbar.ax.axvline(x=1.0, color='gray', linestyle='--')
    cbar.ax.text(0.5, 1.1, "S2'", color='black', ha='center', va='bottom', fontsize=12, transform=cbar.ax.transAxes)

    # 调整布局以防止图例遮挡
    plt.tight_layout(rect=[0, 0, 1, 0.95])


    # Save the figure
    plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','iterations_sorted_1205_4')
    #plot_dir = os.path.join('E:/ccdt_data', 'd_prime_squared','iterations')

    os.makedirs(plot_dir, exist_ok=True)  # 创建目录（如果不存在）

    plot_save_path = os.path.join(plot_dir, f'{NAME}_dp2_ietrations.png')

    # Save the plot
    plt.savefig(plot_save_path)

    print(f' d_prime_squared curves saved at {plot_save_path}')

    return   d2s_all




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
        ds_squared = main(NAME)
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