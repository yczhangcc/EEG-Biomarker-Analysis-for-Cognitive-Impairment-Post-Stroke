import mne
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt
from mne import io
from mne.minimum_norm import read_inverse_operator, source_induced_power
from mne.time_frequency import tfr_morlet
import neurokit2 as nk

subjects = ['MCI01', 'MCI02', 'MCI03'] # with the actual participant name
# 创建一个空列表来存储每个被试的数据
data_list = []

# 循环处理每个被试的数据
for sub in subjects:
    raw = mne.io.read_raw_eeglab(f"D:\mnepythonEEG\datasetqw\{sub}.set")
    raw.load_data()

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

# 指定图片保存路径
import  os
#sub="mci01"
#picturename=f'psd_all_{subnumber}.png'
figure_save_path = r"D:\mnepythonEEG\mod_picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建

# 绘制各通道的功率谱密度
raw_removed.plot_psd(fmin=1., fmax=120.)
#plt.show()
for sub in subjects:
    psdallname=f'psd_all_{sub}.png'
    plt.savefig(os.path.join(figure_save_path , psdallname))

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
out_pca = nk.microstates_segment(raw_removed, method='pca', standardize_eeg=True,n_microstates=5)
nk.microstates_plot(out_pca, gfp=out_pca["GFP"][0:500],epoch=(0,130))
#save picture
for sub in subjects:
    microname=f'microstate_{sub}.png'
    plt.savefig(os.path.join(figure_save_path , microname))

from PIL import Image
# 选择特定频段的数据
freq_band = (50, 85)
raw_removed.filter(*freq_band, picks='eeg')
# 计算频段内的平均功率或能量
freq_band_power = np.mean(np.abs(raw_removed._data) ** 2, axis=-1)  # 这里计算的是平均功率
# 创建一个Info对象，包括位置信息
info = raw_removed.info
# 绘制特定频段的脑地形图
fig2, ax = plt.subplots(1, 1, figsize=(5, 5))
mne.viz.plot_topomap(freq_band_power, info, names=info['ch_names'],cmap='viridis',axes=ax)
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

#最后plt可以一次性画多张图片
plt.show()