import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# 读取Excel表格
excel_path = 'D:\mnepythonEEG\sperate power\powerall.xlsx' # 替换成你的Excel文件路径
df = pd.read_excel(excel_path, index_col=0)  # 假设第一列是索引列

# 提取需要的数据列和行
selected_columns = ['delta', 'theta','alpha', 'beta', 'gamma']
selected_rows = ['mci','mod','sev', 'nc']
data = df.loc[selected_rows, selected_columns]

# 转置数据，使得每列对应一个组
data = data.transpose()

# 重置索引，以便小提琴图能够正确识别横轴
data.reset_index(inplace=True)

# Melt数据，以适应seaborn的小提琴图格式
melted_data = pd.melt(data, id_vars='index', var_name='Group', value_name='Average Power')
# Create a figure and axis
#fig, ax = plt.subplots(figsize=(10, 6))
sns.catplot(x='Group', y='Average Power', hue='index', kind='violin', data=melted_data, height=6,
            linewidth=0.5,gap=4,width=0.8,saturation=2,
            aspect=2, inner='stick',bw_adjust=.9,orient="v")
#sns.swarmplot(data=melted_data, x="Group", y="Average Power", size=3)

plt.show()