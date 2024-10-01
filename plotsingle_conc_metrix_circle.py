import mne
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt
from mne import io
from mne.minimum_norm import read_inverse_operator, source_induced_power
from mne.time_frequency import tfr_morlet
import neurokit2 as nk
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


sub="MCI01"

# 循环处理每个被试的数据
raw = mne.io.read_raw_eeglab(f"D:\mnepythonEEG\datasetqw\MCI01.set")
raw.load_data()

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
# 检查删除通道后的通道数量是否正确
print("通道数量:", len(raw_removed.info['ch_names']))
# 选择感兴趣的通道
channels_of_interest = ['Fp1' ,'Fp2', 'F3' ,'F4' ,'C3' ,'C4' ,'P3' ,'P4' ,'O1', 'O2', 'F7',
            'F8', 'T3', 'T4' ,'T5', 'T6', 'Fz' ,'Cz' ,'Pz']
raw_removed = raw.copy().pick_channels(channels_of_interest)
#raw_removed.pick_channels(channels_of_interest)


# 指定图片保存路径
import  os
figure_save_path = r"D:\mnepythonEEG\picture"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建

# 计算信号包络
envelope_data = np.abs(raw_removed.get_data())

# 将每个主题的信号包络数据添加到列表中
#data_list.append(envelope_data)

# 计算功能性连接（信号包络之间的相关性）
correlation_matrix = np.corrcoef(envelope_data)

import seaborn as sns

# 创建一个函数来绘制连接图
def plot_connectivity_circle(correlation_matrix, channel_labels):
    num_channels = len(channel_labels)
    plt.figure(figsize=(8, 8))

    # 使用Seaborn的heatmap绘制相关性矩阵，颜色表示连接强度
    sns.heatmap(correlation_matrix,annot_kws={"size": 8}, annot=True, xticklabels=channel_labels, yticklabels=channel_labels, cmap="coolwarm")

    plt.title("Connectivity Matrix")
    plt.show()


# 调用函数来绘制连接图
plot_connectivity_circle(correlation_matrix, channels_of_interest)
connection_metrixname=f'connection_metrix_{sub}.png'
plt.savefig(os.path.join(figure_save_path , connection_metrixname))

def plot_custom_connectivity_circle(correlation_matrix, channel_labels):
    num_channels = len(channel_labels)

    # 创建一个无向图对象
    G = nx.Graph()

    # 添加节点（脑区）
    for label in channel_labels:
        G.add_node(label)

    # 添加边（连接）并设置权重
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            weight = correlation_matrix[i, j]
            G.add_edge(channel_labels[i], channel_labels[j], weight=weight)

    # 设置节点位置，可以使用circular_layout或其他布局
    pos = nx.circular_layout(G)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)

    # 绘制边并使用权重来调整线的粗细和颜色
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    # 计算颜色映射
    color_map = plt.get_cmap("coolwarm")
    edge_colors = [color_map(w) for w in weights]
    edge_widths = [w * 3 for w in weights]  # 调整线的粗细
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors)

    # 添加标签
    labels = {label: label for label in channel_labels}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')

    plt.title("Custom Connectivity Circle")
    plt.axis('off')
    plt.show()
plot_custom_connectivity_circle(correlation_matrix, channels_of_interest)
connectivity_circlename=f'connection_circle_{sub}.png'
plt.savefig(os.path.join(figure_save_path , connectivity_circlename))
'''
# 创建一个函数来绘制连接图
def plot_custom_connectivity_circle(correlation_matrix, channel_labels):
    num_channels = len(channel_labels)

    # 创建一个无向图对象
    G = nx.Graph()

    # 添加节点（脑区）
    for label in channel_labels:
        G.add_node(label)

    # 添加边（连接）并设置权重
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            weight = correlation_matrix[i, j]
            G.add_edge(channel_labels[i], channel_labels[j], weight=weight)

    # 设置节点位置，可以使用circular_layout或其他布局
    pos = nx.circular_layout(G)

    # 绘制节点，使用灰色
    node_colors = 'gray'
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)

    # 绘制边并使用权重来调整线的粗细和颜色，使用莫蓝迪色
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    # 创建自定义的颜色映射，根据权重值从小到大，从蓝色到紫色
    norm = plt.Normalize(min(weights), max(weights))
    edge_colors = [cmap(norm(w)) for w in weights]
    edge_widths = [w * 4 for w in weights]  # 增强线的粗细对比

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors)

    # 添加标签
    labels = {label: label for label in channel_labels}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')

    plt.title("Custom Connectivity Circle")
    plt.axis('off')
    plt.show()

    # 调用函数来绘制自定义连接图
plot_custom_connectivity_circle(correlation_matrix, channels_of_interest)
'''
# 如果需要，保存连接矩阵
#np.savetxt('correlation_matrix.csv', correlation_matrix, delimiter=',')

#最后plt可以一次性画多张图片
plt.show()