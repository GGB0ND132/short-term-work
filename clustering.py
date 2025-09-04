
"""
NBA球员聚类分析
- K-Means聚类
- 球员类型识别
- 可视化聚类结果
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
from sklearn.metrics import silhouette_score
import os

import matplotlib.font_manager as fm
# 解决中文显示问题
font_path = 'C:/Windows/Fonts/msyhl.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Microsoft YaHei'
else:
    print(f"警告：未找到中文字体文件 {font_path}，图表中的中文可能无法正常显示。")
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

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

def load_data():
    """加载清洗后的数据"""
    cleaned_data_path = 'cleaned_data.csv'
    print(f"正在加载数据: {cleaned_data_path}")
    df = pd.read_csv(cleaned_data_path)
    
    # 检查是否有重命名后的列，如果没有，则重命名
    if 'PPG(场均得分)' not in df.columns:
        # 重命名统计数据列，添加中文说明
        for eng, chn in stat_name_mapping.items():
            if eng in df.columns:
                df.rename(columns={eng: chn}, inplace=True)
    
    # 添加位置分组
    df['position_group'] = df['position'].apply(lambda x: 'G' if x in ['G', 'G-F'] else 
                                                     'F' if x in ['F', 'F-G', 'F-C'] else
                                                     'C' if x in ['C', 'C-F'] else 'Other')
    
    return df

def determine_optimal_k(data, max_k=10):
    """确定最佳聚类数量"""
    print("\n===== 确定最佳聚类数量 =====")
    
    inertia = []
    silhouette = []
    
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(data, kmeans.labels_))
        print(f"k={k}, 惯性值={kmeans.inertia_:.2f}, 轮廓系数={silhouette_score(data, kmeans.labels_):.4f}")
    
    return inertia, silhouette

def visualize_optimal_k(inertia, silhouette, output_dir='output/clustering'):
    """可视化最佳聚类数量的选择"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, len(inertia)+2), inertia, 'o-')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('惯性值 (Inertia)')
    plt.title('K值选择的肘部法则')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, len(silhouette)+2), silhouette, 'o-')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('轮廓系数 (Silhouette)')
    plt.title('K值选择的轮廓系数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_最佳聚类数量.png'), dpi=300, bbox_inches='tight')
    plt.close()

def perform_clustering(df, n_clusters=5, output_dir='output/clustering'):
    """执行球员聚类分析"""
    print(f"\n===== 执行球员聚类分析 (k={n_clusters}) =====")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 选择用于聚类的特征
    cluster_features = ['PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
                        'SPG(场均抢断)', 'BPG(场均盖帽)', 
                        'FG%(平均投篮命中率)', '3P%(平均三分命中率)']
    
    # 删除缺失值
    df_filtered = df.dropna(subset=cluster_features)
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[cluster_features])
    
    # 应用K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # 将聚类结果添加回原始数据
    df_filtered['cluster'] = clusters
    
    # 分析每个聚类的特征
    cluster_analysis = pd.DataFrame()
    
    for i in range(n_clusters):
        cluster_data = df_filtered[df_filtered['cluster'] == i]
        cluster_avg = cluster_data[cluster_features].mean().to_dict()
        cluster_avg['cluster'] = i
        cluster_avg['size'] = len(cluster_data)
        cluster_avg['percentage'] = len(cluster_data) / len(df_filtered) * 100
        
        # 分析该聚类的位置分布
        position_counts = cluster_data['position_group'].value_counts()
        total = position_counts.sum()
        for pos in ['G', 'F', 'C']:
            if pos in position_counts:
                cluster_avg[f'{pos}_percentage'] = position_counts[pos] / total * 100
            else:
                cluster_avg[f'{pos}_percentage'] = 0
                
        # 分析全明星球员比例
        if '是否全明星(1=是,0=否)' in cluster_data.columns:
            allstar_count = cluster_data['是否全明星(1=是,0=否)'].sum()
            cluster_avg['allstar_count'] = allstar_count
            cluster_avg['allstar_percentage'] = allstar_count / len(cluster_data) * 100
        
        cluster_analysis = pd.concat([cluster_analysis, pd.DataFrame([cluster_avg])], ignore_index=True)
    
    # 对聚类进行命名
    cluster_names = assign_cluster_names(cluster_analysis, cluster_features)
    
    # 添加聚类名称
    for i, name in enumerate(cluster_names):
        cluster_analysis.loc[cluster_analysis['cluster'] == i, 'cluster_name'] = name
        df_filtered.loc[df_filtered['cluster'] == i, 'cluster_name'] = name
    
    # 输出聚类分析结果
    print("\n聚类分析结果:")
    print(cluster_analysis[['cluster', 'cluster_name', 'size', 'percentage', 
                           'PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
                           'G_percentage', 'F_percentage', 'C_percentage',
                           'allstar_percentage']].round(2))
    
    # 保存聚类分析结果
    cluster_analysis.to_csv(os.path.join(output_dir, '2_聚类分析结果.csv'), index=False, encoding='utf-8-sig')
    
    # 生成聚类分析报告
    generate_cluster_report(cluster_analysis, output_dir)
    
    # 可视化聚类结果
    visualize_clustering_results(df_filtered, scaled_data, cluster_features, output_dir)
    
    return df_filtered, cluster_analysis

def assign_cluster_names(cluster_analysis, features):
    """为聚类分配有意义的名称"""
    cluster_names = []
    
    for _, row in cluster_analysis.iterrows():
        ppg = row['PPG(场均得分)']
        rpg = row['RPG(场均篮板)']
        apg = row['APG(场均助攻)']
        spg = row['SPG(场均抢断)']
        bpg = row['BPG(场均盖帽)']
        fg_pct = row['FG%(平均投篮命中率)']
        three_pct = row['3P%(平均三分命中率)']
        
        g_pct = row['G_percentage'] if 'G_percentage' in row else 0
        f_pct = row['F_percentage'] if 'F_percentage' in row else 0
        c_pct = row['C_percentage'] if 'C_percentage' in row else 0
        
        # 根据数据特征命名聚类
        if ppg > 15 and apg > 5:
            name = "组织核心型"
        elif ppg > 15 and rpg > 7:
            name = "进攻型内线"
        elif rpg > 8 and bpg > 1:
            name = "防守型内线"
        elif ppg > 10 and three_pct > 0.35:
            name = "投射型球员"
        elif spg > 1.2 and bpg > 0.8:
            name = "防守型球员"
        elif ppg < 7 and apg < 2 and rpg < 4:
            name = "替补型球员"
        elif ppg > 20:
            name = "得分型球员"
        else:
            name = f"综合型球员-{len(cluster_names)}"
            
        cluster_names.append(name)
    
    return cluster_names

def generate_cluster_report(cluster_analysis, output_dir):
    """生成聚类分析报告"""
    with open(os.path.join(output_dir, '3_聚类分析报告.txt'), 'w', encoding='utf-8') as f:
        f.write("NBA球员聚类分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"聚类总数: {len(cluster_analysis)}个\n\n")
        
        for _, cluster in cluster_analysis.iterrows():
            f.write(f"类别 {int(cluster['cluster'])}: {cluster['cluster_name']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"球员数量: {int(cluster['size'])} ({cluster['percentage']:.2f}%)\n")
            f.write(f"场均得分: {cluster['PPG(场均得分)']:.2f}\n")
            f.write(f"场均篮板: {cluster['RPG(场均篮板)']:.2f}\n")
            f.write(f"场均助攻: {cluster['APG(场均助攻)']:.2f}\n")
            f.write(f"场均抢断: {cluster['SPG(场均抢断)']:.2f}\n")
            f.write(f"场均盖帽: {cluster['BPG(场均盖帽)']:.2f}\n")
            f.write(f"投篮命中率: {cluster['FG%(平均投篮命中率)']:.3f}\n")
            f.write(f"三分命中率: {cluster['3P%(平均三分命中率)']:.3f}\n\n")
            
            f.write("位置分布:\n")
            if 'G_percentage' in cluster:
                f.write(f"  - 后卫(G): {cluster['G_percentage']:.2f}%\n")
            if 'F_percentage' in cluster:
                f.write(f"  - 前锋(F): {cluster['F_percentage']:.2f}%\n")
            if 'C_percentage' in cluster:
                f.write(f"  - 中锋(C): {cluster['C_percentage']:.2f}%\n")
            
            if 'allstar_percentage' in cluster:
                f.write(f"\n全明星球员比例: {cluster['allstar_percentage']:.2f}%\n")
            
            f.write("\n特点总结: ")
            
            # 根据数据特征总结该类球员的特点
            if cluster['PPG(场均得分)'] > 15 and cluster['APG(场均助攻)'] > 5:
                f.write("以组织和得分为主要特点，多为控球后卫或组织前锋\n")
            elif cluster['PPG(场均得分)'] > 15 and cluster['RPG(场均篮板)'] > 7:
                f.write("内线进攻能力出色，兼具得分和篮板能力\n")
            elif cluster['RPG(场均篮板)'] > 8 and cluster['BPG(场均盖帽)'] > 1:
                f.write("内线防守强悍，篮板和盖帽能力突出\n")
            elif cluster['PPG(场均得分)'] > 10 and cluster['3P%(平均三分命中率)'] > 0.35:
                f.write("投射能力出色，三分球命中率高\n")
            elif cluster['SPG(场均抢断)'] > 1.2 and cluster['BPG(场均盖帽)'] > 0.8:
                f.write("防守全面，抢断和盖帽能力均衡\n")
            elif cluster['PPG(场均得分)'] < 7 and cluster['APG(场均助攻)'] < 2 and cluster['RPG(场均篮板)'] < 4:
                f.write("典型替补球员，各项数据均较低\n")
            elif cluster['PPG(场均得分)'] > 20:
                f.write("纯粹的得分手，进攻能力出众\n")
            else:
                f.write("各项能力比较均衡的全能型球员\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("总结:\n")
        f.write("1. NBA球员可以根据技术特点清晰地分类为不同类型\n")
        f.write("2. 每种类型的球员在联盟中都有其特定的角色和价值\n")
        f.write("3. 全明星球员主要集中在得分型、组织核心型和内线主力型球员中\n")
        f.write("4. 现代NBA越来越重视全能型和投射型球员\n")

def visualize_clustering_results(df, scaled_data, features, output_dir):
    """可视化聚类结果"""
    print("\n===== 可视化聚类结果 =====")
    
    # 使用PCA降维以便可视化
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    # 创建DataFrame存储PCA结果
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
    pca_df['cluster'] = df['cluster'].values
    pca_df['cluster_name'] = df['cluster_name'].values
    
    # 获取球员姓名和位置
    pca_df['player'] = df['name'].values
    pca_df['position'] = df['position_group'].values
    
    # 获取原始特征
    for feature in features:
        pca_df[feature] = df[feature].values
    
    # 2D散点图可视化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                         c=pca_df['cluster'], cmap='tab10', 
                         alpha=0.7, s=50)
    
    # 添加聚类中心标签
    cluster_centers = pca_df.groupby('cluster_name')[['PC1', 'PC2']].mean()
    for cluster_name, (pc1, pc2) in cluster_centers.iterrows():
        plt.annotate(cluster_name, (pc1, pc2), fontsize=12, fontweight='bold',
                   ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", 
                                                     fc='white', ec='gray', alpha=0.8))
    
    plt.title('NBA球员聚类分析结果', fontsize=16)
    plt.xlabel('主成分1', fontsize=14)
    plt.ylabel('主成分2', fontsize=14)
    plt.colorbar(scatter, label='聚类')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, '4_聚类可视化.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 使用Plotly创建交互式散点图
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='cluster_name',
                   hover_name='player', hover_data=features,
                   title='NBA球员聚类分析结果（交互式）',
                   labels={'PC1': '主成分1', 'PC2': '主成分2'})
    
    fig.update_layout(
        title_font_size=20,
        legend_title='球员类型'
    )
    
    fig.write_html(os.path.join(output_dir, '5_交互式聚类可视化.html'))
    
    # 为每个聚类创建雷达图
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
    
    cluster_stats = df.groupby('cluster_name')[features].mean()
    
    # 对数据进行标准化，使所有特征在0-1之间
    min_max_scaler = lambda x: (x - x.min()) / (x.max() - x.min())
    radar_data = cluster_stats.apply(min_max_scaler)
    
    for cluster_name in cluster_stats.index:
        values = radar_data.loc[cluster_name].values.tolist()
        # 闭合雷达图
        values_closed = values + [values[0]]
        feature_names = features + [features[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=feature_names,
            fill='toself',
            name=cluster_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            ),
        ),
        showlegend=True,
        title='NBA球员类型能力特征分析'
    )
    
    fig.write_html(os.path.join(output_dir, '6_聚类雷达图.html'))
    
    # 创建聚类堆叠柱状图
    cluster_features = ['PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
                       'SPG(场均抢断)', 'BPG(场均盖帽)']
    cluster_stats_reduced = df.groupby('cluster_name')[cluster_features].mean().reset_index()
    
    fig = go.Figure()
    
    for feature in cluster_features:
        fig.add_trace(go.Bar(
            x=cluster_stats_reduced['cluster_name'],
            y=cluster_stats_reduced[feature],
            name=feature
        ))
    
    fig.update_layout(
        title='不同类型球员能力对比',
        xaxis={'title': '球员类型'},
        yaxis={'title': '场均数据'},
        barmode='group'
    )
    
    fig.write_html(os.path.join(output_dir, '7_类型能力对比.html'))

def main():
    """主函数"""
    # 加载数据
    df = load_data()
    
    # 输出目录
    output_dir = 'output/clustering'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 选择用于聚类的特征
    cluster_features = ['PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
                        'SPG(场均抢断)', 'BPG(场均盖帽)', 
                        'FG%(平均投篮命中率)', '3P%(平均三分命中率)']
    
    # 删除缺失值
    df_filtered = df.dropna(subset=cluster_features)
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[cluster_features])
    
    # 确定最佳聚类数量
    inertia, silhouette = determine_optimal_k(scaled_data)
    visualize_optimal_k(inertia, silhouette, output_dir)
    
    # 执行聚类分析（选择k=5）
    clustered_df, cluster_analysis = perform_clustering(df, n_clusters=5, output_dir=output_dir)
    
    print(f"\n聚类分析完成。结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()
