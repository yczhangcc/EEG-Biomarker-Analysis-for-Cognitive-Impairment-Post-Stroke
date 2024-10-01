import mne
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt
from mne import io
from mne.minimum_norm import read_inverse_operator, source_induced_power
from mne.time_frequency import tfr_morlet
import neurokit2 as nk
from neurokit2.microstates import microstates_peaks
import pandas as pd

#subjects = ['MCI01', 'MCI02', 'MCI03','MCI04', 'MCI05', 'MCI06','MCI07', 'MCI08'] # with the actual participant name
#subjects = ['sev01', 'sev02', 'sev03','sev09', 'sev04', 'sev06','sev07', 'sev08']
#subjects = [ 'NC05', 'NC06','NC07', 'NC08']#'NC01', 'NC02', 'NC03','NC04',
#subjects = ['mild_1_remove', 'mild_2_remove', 'mild_3_remove','mild_4_remove', 'mild_5_remove', 'mild_6_remove','mild_7_remove', 'mild_8_remove']
subjects = ['mod_1_REMOVE', 'mod_2_REMOVE', 'mod_3_REMOVE','mod_4_REMOVE', 'mod_5_REMOVE', 'mod_6_REMOVE','mod_7_REMOVE', 'mod_8_REMOVE']
group="mod"
band="gamma"
# 创建一个空列表来存储每个被试的数据
data_list = []
# 指定图片保存路径
import  os
#sub="mci01"
#picturename=f'psd_all_{subnumber}.png'
figure_save_path = r"D:\mnepythonEEG\picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建

# 循环处理每个被试的数据
for sub in subjects:
    raw = mne.io.read_raw_eeglab(f"D:\mnepythonEEG\datasetqw\{sub}.set")
    raw.load_data()
    # 设置采样率
    #sampling_rate = 500  # 根据实际采样率进行设置
    # 使用 microstates_peaks() 函数，并提供采样率
    #peaks = nk.microstates_peaks(raw,sampling_rate=sampling_rate)
#raw = mne.io.read_raw_eeglab(r"D:\mnepythonEEG\datasetqw\{subject}.set")
#sub="mod08"
#raw.load_data()  # 加载数据到内存

# 需要一个电极名
    ch_names = ['Fp1' 'Fp2' 'F3' 'F4' 'C3' 'C4' 'P3' 'P4' 'O1' 'O2' 'F7'
            'F8' 'T3' 'T4' 'T5' 'T6' 'Fz' 'Cz' 'Pz'
            'POL E''POL PG1''POL PG2' 'POL T1''POLT2''POLX1''POLX2'
            'POLX3''POLX4''POLX5''POLX6''POLX7' 'POLSpO2''POLEtCO2'
            'POLDC03''POLDC04''POLDC05' 'POLDC06''POLPulse''POLCO2Wave''POL$A1''POL$A2']

# 电极信息，有俩种用法一个是导入自己的电极图，一个是引用官方的图,这里选择导入自己的locs文件
    locs_info_path = "Standard-10-20-Cap44-1.locs"
# 这里根据读取的电极名字要和edf读取的电极名一致，否则需要修改
    new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
    old_chan_names = raw.info["ch_names"]
    chan_names_dict = {old_chan_names[i]: new_chan_names[i] for i in range(41)}
# 更新数据的电极名字（不是标准名无法运行）
    raw.rename_channels(chan_names_dict)

#remove掉没有位置信息的POL电极数据，注意后面要用raw_remove画图
    raw_removed = raw.copy().drop_channels(['POLE','POLPG1','POLPG2' ,'POLT1','POLT2','POLX1','POLX2',
            'POLX3','POLX4','POLX5','POLX6','POLX7', 'POLSpO2','POLEtCO2',
            'POLDC03','POLDC04','POLDC05', 'POLDC06','POLPulse','POLCO2Wave','POL$A1','POL$A2'])
    # 插值以确保所有数据具有相同的通道数量
    if len(raw_removed.info['ch_names']) < len(raw_removed.info['ch_names']):
        raw_removed.interpolate_bads()
    # 将原始数据添加到数据列表
    data_list.append(raw_removed.get_data())
# 获取数据列表中最短的数据的长度
min_data_length = min([data.shape[1] for data in data_list])

# 截断或填充数据，使它们具有相同的形状
data_list = [data[:, :min_data_length] if data.shape[1] > min_data_length else np.hstack((data, np.zeros((data.shape[0], min_data_length - data.shape[1])))) for data in data_list]
# 计算多个被试的平均数据
# Calculate average power for each raw object in data_list,计算power公式如下图所示。然后，讲各个组的被试进行平均，分别得到每个组19个通道的平均值。
power_list = [np.mean(np.abs(raw) ** 2, axis=-1) for raw in data_list]

# Assuming the last dimension is time, you may need to adjust if it's different
grand_average_data = np.mean(np.array(power_list), axis=0)[:, np.newaxis]
average_value = np.mean(grand_average_data)
print(grand_average_data)


# 创建一个 RawArray 对象
info = raw_removed.info
raw_combined = mne.io.RawArray(grand_average_data, info)


#Delta
#freq_band = (0.1, 4)
#Theta波: 4Hz-8Hz
#freq_band = (4, 8)
#Alpha波: 8Hz-12Hz
#freq_band = (8, 12)
#Beta 波: 12Hz-30Hz
#freq_band = (12, 30)
#Gamma波: 30Hz-100Hz
freq_band = (30, 100)
raw_combined.filter(*freq_band, picks='eeg')
average_value1 = np.mean(grand_average_data)
print(grand_average_data)

matrix_list=grand_average_data
# Reshape matrices to 1D arrays
matrix_list = [matrix.flatten() for matrix in matrix_list]
# 创建一个字典，将矩阵转换为DataFrame
#data_dict = {f'Matrix_{i + 1}': pd.DataFrame(matrix) for i, matrix in enumerate(matrix_list)}
# 将矩阵列表转换为 DataFrame
df = pd.DataFrame(matrix_list)
# 将DataFrame写入Excel文件

excel_path = fr"D:\mnepythonEEG\{group}_{band}.xlsx"  # 替换成你想要保存的Excel文件路径
# 将 DataFrame 保存到 Excel 文件中
df.to_excel(excel_path, index="nc", header={band})
