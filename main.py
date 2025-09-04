
"""
NBA球员数据分析与可视化
- 加载清洗后的数据
- 描述性统计分析
- 可视化分析
- 机器学习分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from datetime import datetime

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子，保证结果可重现
np.random.seed(42)

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 设置可视化样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(style="darkgrid")

# 中英文字段映射
stat_name_mapping = {
    'PPG': 'PPG(场均得分)',
    'RPG': 'RPG(场均篮板)',
    'APG': 'APG(场均助攻)',
    'SPG': 'SPG(场均抢断)',
    'BPG': 'BPG(场均盖帽)',
    'Efficiency': '效率值',
    'Total_Points': '总得分',
    'Total_Rebounds': '总篮板',
    'Total_Assists': '总助攻',
    'Total_Steals': '总抢断',
    'Total_Blocks': '总盖帽',
    'Games_Played': '出场次数',
    'height_inches': '身高(英寸)',
    'weight_lbs': '体重(磅)',
    'career_length': '生涯长度(年)',
    'birth_year': '出生年份',
    'Avg_PER': 'PER(平均效率值)',
    'Total_Win_Shares': 'WS(总胜利贡献值)',
    'Avg_FG_Pct': 'FG%(平均投篮命中率)',
    'Avg_3P_Pct': '3P%(平均三分命中率)',
    'Avg_FT_Pct': 'FT%(平均罚球命中率)',
    'Is_AllStar': '是否全明星(1=是,0=否)'
}

# 定义数据文件路径
CLEANED_DATA_PATH = 'cleaned_data.csv'

def load_cleaned_data():
    """
    加载清洗后的数据
    """
    print("正在加载清洗后的数据...")
    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"错误：找不到清洗后的数据文件 {CLEANED_DATA_PATH}")
        print("请先运行 data_cleaning.py 生成清洗后的数据")
        return None
    
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # 检查是否有重命名后的列，如果没有，则重命名
    if 'PPG(场均得分)' not in df.columns:
        # 重命名统计数据列，添加中文说明
        for eng, chn in stat_name_mapping.items():
            if eng in df.columns:
                df.rename(columns={eng: chn}, inplace=True)
    
    print(f"数据形状: {df.shape}")
    return df

def descriptive_statistics(df):
    """
    描述性统计分析
    """
    print("\n===== 描述性统计分析 =====")
    
    # 数值型特征的描述性统计
    numeric_columns = ['height_inches', 'weight_lbs', 'birth_year', 'career_length',
                      'Games_Played', 'Total_Points', 'PPG', 'RPG', 'APG', 
                      'SPG', 'BPG', 'Efficiency', 'Is_AllStar']
    
    stats_df = df[numeric_columns].describe()
    print("\n数值型特征描述性统计:")
    print(stats_df)
    
    # 统计全明星球员比例
    allstar_count = df['Is_AllStar'].sum()
    allstar_pct = (allstar_count / len(df)) * 100
    print(f"\n全明星球员数量: {allstar_count} ({allstar_pct:.2f}%)")
    
    # 按位置分组统计
    print("\n按位置分组的场均数据:")
    position_stats = df.groupby('position').agg({
        'PPG': 'mean',
        'RPG': 'mean',
        'APG': 'mean',
        'SPG': 'mean',
        'BPG': 'mean',
        'Efficiency': 'mean',
        'Games_Played': 'mean',
        'height_inches': 'mean',
        'weight_lbs': 'mean'
    }).round(2)
    
    print(position_stats)
    
    return stats_df, position_stats

def visualization_analysis(df):
    """
    数据可视化分析
    """
    print("\n===== 数据可视化分析 =====")
    
    # 创建输出目录
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 球员身高分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['height_inches'].dropna(), kde=True, bins=20)
    plt.title('NBA球员身高分布')
    plt.xlabel('身高(英寸)')
    plt.ylabel('球员数量')
    plt.savefig(os.path.join(output_dir, '01_height_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 2. 球员体重分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['weight_lbs'].dropna(), kde=True, bins=20)
    plt.title('NBA球员体重分布')
    plt.xlabel('体重(磅)')
    plt.ylabel('球员数量')
    plt.savefig(os.path.join(output_dir, '02_weight_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 3. 身高与体重的关系
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='height_inches', y='weight_lbs', hue='position', alpha=0.7)
    plt.title('NBA球员身高与体重的关系')
    plt.xlabel('身高(英寸)')
    plt.ylabel('体重(磅)')
    plt.savefig(os.path.join(output_dir, '03_height_weight_relationship.png'), dpi=300, bbox_inches='tight')
    
    # 4. 不同位置的得分能力
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='position', y='PPG')
    plt.title('不同位置球员的场均得分')
    plt.xlabel('位置')
    plt.ylabel('场均得分(PPG)')
    plt.savefig(os.path.join(output_dir, '04_position_ppg_boxplot.png'), dpi=300, bbox_inches='tight')
    
    # 5. 生涯长度与效率值的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='career_length', y='Efficiency', hue='Is_AllStar', alpha=0.7)
    plt.title('生涯长度与效率值的关系')
    plt.xlabel('生涯长度(年)')
    plt.ylabel('效率值')
    plt.savefig(os.path.join(output_dir, '05_career_efficiency.png'), dpi=300, bbox_inches='tight')
    
    # 6. 雷达图：不同位置球员能力对比
    # 使用plotly绘制
    position_stats = df.groupby('position').agg({
        'PPG': 'mean',
        'RPG': 'mean',
        'APG': 'mean',
        'SPG': 'mean',
        'BPG': 'mean',
    }).reset_index()
    
    fig = go.Figure()
    
    categories = ['PPG', 'RPG', 'APG', 'SPG', 'BPG']
    for i, pos in enumerate(position_stats['position']):
        values = position_stats.loc[i, categories].values.tolist()
        # 闭合雷达图
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name=pos
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            ),
        ),
        showlegend=True,
        title='不同位置球员能力对比'
    )
    
    fig.write_html(os.path.join(output_dir, '06_position_skills_radar.html'))
    
    # 7. 热图：各项能力指标相关性
    corr_columns = ['height_inches', 'weight_lbs', 'career_length', 
                   'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'Efficiency']
    corr_matrix = df[corr_columns].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('NBA球员各项能力指标相关性')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    
    print(f"可视化图表已保存到 {output_dir} 目录")

def machine_learning_analysis(df):
    """
    机器学习分析：球员聚类
    """
    print("\n===== 机器学习分析 =====")
    
    # 创建输出目录
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 选择用于聚类的特征
    cluster_features = ['PPG', 'RPG', 'APG', 'SPG', 'BPG']
    
    # 删除缺失值
    cluster_data = df[cluster_features].dropna()
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # 使用PCA降维以便可视化
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    # 创建一个DataFrame来存储PCA结果
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
    
    # 确定最佳聚类数（这里简化为使用5个聚类，实际应该使用肘部法则）
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # 将聚类结果添加到PCA数据中
    pca_df['Cluster'] = clusters
    
    # 添加原始特征和球员信息到PCA数据
    for feature in cluster_features:
        pca_df[feature] = cluster_data.reset_index()[feature]
    
    # 添加球员姓名（如果有）
    if 'name' in df.columns:
        pca_df['Player'] = df.loc[cluster_data.index, 'name'].values
    
    # 可视化聚类结果
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
    plt.title('NBA球员聚类分析')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.savefig(os.path.join(output_dir, '08_player_clustering.png'), dpi=300, bbox_inches='tight')
    
    # 分析每个聚类的特征
    cluster_analysis = pd.DataFrame()
    
    for i in range(k):
        cluster_data = pca_df[pca_df['Cluster'] == i]
        cluster_avg = cluster_data[cluster_features].mean().to_dict()
        cluster_avg['Cluster'] = i
        cluster_avg['Size'] = len(cluster_data)
        cluster_avg['Percentage'] = len(cluster_data) / len(pca_df) * 100
        cluster_analysis = pd.concat([cluster_analysis, pd.DataFrame([cluster_avg])], ignore_index=True)
    
    print("\n球员聚类分析结果:")
    print(cluster_analysis.round(2))
    
    # 可视化每个聚类的特征雷达图
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
    
    for i in range(k):
        cluster_values = cluster_analysis.loc[cluster_analysis['Cluster'] == i, cluster_features].values[0].tolist()
        # 闭合雷达图
        cluster_values_closed = cluster_values + [cluster_values[0]]
        categories_closed = cluster_features + [cluster_features[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=cluster_values_closed,
            theta=categories_closed,
            fill='toself',
            name=f'集群 {i}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            ),
        ),
        showlegend=True,
        title='球员类型特征分析'
    )
    
    fig.write_html(os.path.join(output_dir, '09_cluster_radar_chart.html'))
    
    return cluster_analysis

def run_all_analyses():
    """
    运行所有分析
    """
    import data_cleaning
    import statistical_analysis
    import visualization
    import clustering
    import prediction
    
    print("\n===== NBA球员数据分析系统 =====")
    print("1. 数据清洗与特征创建")
    print("2. 统计分析")
    print("3. 数据可视化")
    print("4. 球员聚类分析")
    print("5. 预测分析")
    print("6. 运行所有分析")
    print("0. 退出")
    
    choice = input("\n请选择要运行的功能 (0-6): ")
    
    if choice == '1':
        print("\n开始执行数据清洗...")
        data_cleaning.main()
    elif choice == '2':
        print("\n开始执行统计分析...")
        statistical_analysis.main()
    elif choice == '3':
        print("\n开始执行数据可视化...")
        visualization.main()
    elif choice == '4':
        print("\n开始执行球员聚类分析...")
        clustering.main()
    elif choice == '5':
        print("\n开始执行预测分析...")
        prediction.main()
    elif choice == '6':
        print("\n开始执行所有分析...")
        data_cleaning.main()
        statistical_analysis.main()
        visualization.main()
        clustering.main()
        prediction.main()
        print("\n所有分析已完成！")
    elif choice == '0':
        print("退出程序")
        return
    else:
        print("无效选择，请重新运行程序")

def main():
    """
    主函数
    """
    run_all_analyses()

if __name__ == "__main__":
    main()
