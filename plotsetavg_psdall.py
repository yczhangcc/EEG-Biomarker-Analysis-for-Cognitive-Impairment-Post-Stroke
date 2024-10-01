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
level="mod"
#subjects = ['MCI01', 'MCI02', 'MCI03','MCI04', 'MCI05', 'MCI06','MCI07', 'MCI08'] # with the actual participant name
#subjects = ['sev01', 'sev02', 'sev03','sev09', 'sev04', 'sev06','sev07', 'sev08']
#subjects = [ 'NC05', 'NC06','NC07', 'NC08']#'NC01', 'NC02', 'NC03','NC04',
#subjects = ['mild_1_remove', 'mild_2_remove', 'mild_3_remove','mild_4_remove', 'mild_5_remove', 'mild_6_remove','mild_7_remove', 'mild_8_remove']
subjects = ['mod_1_REMOVE', 'mod_2_REMOVE', 'mod_3_REMOVE','mod_4_REMOVE', 'mod_5_REMOVE', 'mod_6_REMOVE','mod_7_REMOVE', 'mod_8_REMOVE']
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
grand_average_data = np.mean(data_list, axis=0)

# 创建一个 RawArray 对象
info = raw_removed.info
raw_combined = mne.io.RawArray(grand_average_data, info)

# 绘制平均 PSD 图
fig = raw_combined.plot_psd(fmin=1, fmax=120)
#plt.title('Grand Average PSD')  # 添加标题


# 绘制各通道的功率谱密度
#grand_average_data.plot_psd(fmin=1., fmax=120.)
#plt.show()
psdallname=f'psd_all_{level}_avg.png'
plt.savefig(os.path.join(figure_save_path , psdallname))
'''
#psd头皮图（分频段）
# 创建一个新的Matplotlib图形
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
fig=raw_removed.plot_psd_topomap(ch_type='eeg', normalize=True,axes=ax)
# 设置标题
fig.suptitle('PSD Topomap')
# 获取图形对象
psd_fig = plt.gcf()
#plt.show()
for sub in subjects:
    psdfrename=f'psd_fre_{sub}.png'
    psd_fig.savefig(os.path.join(figure_save_path , psdfrename))#第一个是指存储路径，第二个是图片名字

# Cluster microstates
#Example with PCA
out_pca = nk.microstates_segment(grand_average_data, method='pca', standardize_eeg=True,n_microstates=5)
nk.microstates_plot(out_pca, gfp=out_pca["GFP"][0:500],epoch=(0,130))
#save picture
for sub in subjects:
    microname=f'microstate_{sub}.png'
    plt.savefig(os.path.join(figure_save_path , microname))


for sub in subjects:
    from PIL import Image
# 选择特定频段的数据
    freq_band = (50, 85)
    raw_removed.filter(*freq_band, picks='eeg')
# 计算频段内的平均功率或能量
    freq_band_power = np.mean(np.abs(raw_removed.get_data()) ** 2, axis=-1)  # 这里计算的是平均功率
    data_list.append(freq_band_power)
    print(data_list)

# 计算多个被试的平均值
grand_average = np.mean(data_list, axis=0)
print(grand_average)

# 创建一个Info对象，包括位置信息
info = raw_removed.info
# 绘制特定频段的脑地形图
fig2, ax = plt.subplots(1, 1, figsize=(5, 5))
mne.viz.plot_topomap(grand_average, info, names=info['ch_names'],cmap='viridis',axes=ax)
# add titles
fig2.suptitle('50-85 Topomap')


# 将Matplotlib图形保存为临时文件
temp_filename = os.path.join(figure_save_path, 'temp_image.png')
fig2.savefig(temp_filename, bbox_inches='tight', pad_inches=0)
# 使用Pillow库打开临时图像并保存
image = Image.open(temp_filename)
for sub in subjects:
    gammaname=f'50_85hz_{sub}.png'
    image.save(os.path.join(figure_save_path, gammaname))
# 删除临时文件
os.remove(temp_filename)
'''

#最后plt可以一次性画多张图片
plt.show()