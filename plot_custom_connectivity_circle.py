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
import seaborn as sns
import os
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