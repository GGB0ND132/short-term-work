#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA球员数据可视化
- 基础图表：直方图、箱线图、散点图、折线图
- 高级图表：雷达图、热力图、条形图
- 交互式图表：使用Plotly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    # 添加年代分组
    def assign_era(year):
        if pd.isna(year):
            return "Unknown"
        elif year < 1960:
            return "1950s"
        elif year < 1970:
            return "1960s"
        elif year < 1980:
            return "1970s"
        elif year < 1990:
            return "1980s"
        elif year < 2000:
            return "1990s"
        elif year < 2010:
            return "2000s"
        else:
            return "2010s+"
    
    df['era'] = df['year_start'].apply(assign_era)
    
    return df

def create_basic_charts(df, output_dir='output/visualizations'):
    """创建基础图表：直方图、箱线图、散点图、折线图"""
    print("\n===== 创建基础图表 =====")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 身高分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='身高(英寸)', kde=True, bins=20)
    plt.title('NBA球员身高分布', fontsize=14)
    plt.xlabel('身高(英寸)')
    plt.ylabel('球员数量')
    plt.savefig(os.path.join(output_dir, '1_身高分布直方图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 不同位置得分箱线图
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='position_group', y='PPG(场均得分)')
    plt.title('不同位置球员场均得分分布', fontsize=14)
    plt.xlabel('位置')
    plt.ylabel('场均得分(PPG)')
    plt.savefig(os.path.join(output_dir, '2_位置得分箱线图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 身高-体重散点图（按位置着色）
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='身高(英寸)', y='体重(磅)', 
                    hue='position_group', size='效率值',
                    sizes=(20, 200), alpha=0.7)
    plt.title('NBA球员身高-体重关系（按位置和效率值）', fontsize=14)
    plt.xlabel('身高(英寸)')
    plt.ylabel('体重(磅)')
    plt.legend(title='位置')
    plt.savefig(os.path.join(output_dir, '3_身高体重散点图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 年代变化折线图
    yearly_stats = df.groupby('year_start').agg({
        'PPG(场均得分)': 'mean',
        'RPG(场均篮板)': 'mean',
        'APG(场均助攻)': 'mean',
        '3P%(平均三分命中率)': 'mean',
    }).reset_index()
    
    # 过滤掉样本太少的年份
    year_counts = df['year_start'].value_counts()
    valid_years = year_counts[year_counts > 5].index
    yearly_stats = yearly_stats[yearly_stats['year_start'].isin(valid_years)]
    
    plt.figure(figsize=(14, 8))
    plt.plot(yearly_stats['year_start'], yearly_stats['PPG(场均得分)'], 'o-', label='场均得分')
    plt.plot(yearly_stats['year_start'], yearly_stats['RPG(场均篮板)'], 's-', label='场均篮板')
    plt.plot(yearly_stats['year_start'], yearly_stats['APG(场均助攻)'], '^-', label='场均助攻')
    
    plt.title('NBA球员数据历史变化趋势', fontsize=14)
    plt.xlabel('年份')
    plt.ylabel('场均数据')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '4_数据历史趋势.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 生涯长度与效率值的关系（按是否全明星着色）
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='生涯长度(年)', y='效率值', 
                    hue='是否全明星(1=是,0=否)', size='出场次数',
                    sizes=(20, 200), alpha=0.7, palette=['blue', 'red'])
    plt.title('NBA球员生涯长度与效率值的关系', fontsize=14)
    plt.xlabel('生涯长度(年)')
    plt.ylabel('效率值')
    plt.legend(title='全明星球员')
    plt.savefig(os.path.join(output_dir, '5_生涯长度效率值散点图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"基础图表已保存到 {output_dir} 目录")

def create_advanced_charts(df, output_dir='output/visualizations'):
    """创建高级图表：雷达图、热力图、条形图"""
    print("\n===== 创建高级图表 =====")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 各项能力指标相关性热力图
    corr_columns = ['PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
                    'SPG(场均抢断)', 'BPG(场均盖帽)', '效率值',
                    'PER(平均效率值)', 'WS(总胜利贡献值)',
                    '身高(英寸)', '体重(磅)', '生涯长度(年)']
    
    corr_matrix = df[corr_columns].corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                fmt='.2f', linewidths=0.5, cbar_kws={'label': '相关系数'})
    plt.title('NBA球员各项能力指标相关性热力图', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_相关性热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 不同位置球员能力对比条形图
    position_stats = df.groupby('position_group').agg({
        'PPG(场均得分)': 'mean',
        'RPG(场均篮板)': 'mean',
        'APG(场均助攻)': 'mean',
        'SPG(场均抢断)': 'mean',
        'BPG(场均盖帽)': 'mean'
    }).reset_index()
    
    metrics = ['PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 'SPG(场均抢断)', 'BPG(场均盖帽)']
    
    plt.figure(figsize=(14, 10))
    
    x = np.arange(len(position_stats['position_group']))
    width = 0.15
    multiplier = 0
    
    for attribute in metrics:
        offset = width * multiplier
        rects = plt.bar(x + offset, position_stats[attribute], width, label=attribute)
        multiplier += 1
    
    plt.xlabel('位置', fontsize=12)
    plt.ylabel('场均数据', fontsize=12)
    plt.title('不同位置球员能力对比', fontsize=14)
    plt.xticks(x + width * 2, position_stats['position_group'])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    
    plt.savefig(os.path.join(output_dir, '7_位置能力条形图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 年代变化条形图
    era_stats = df.groupby('era').agg({
        'PPG(场均得分)': 'mean',
        'RPG(场均篮板)': 'mean',
        'APG(场均助攻)': 'mean',
        '3P%(平均三分命中率)': 'mean',
        '身高(英寸)': 'mean',
        '体重(磅)': 'mean'
    }).reset_index()
    
    # 按年代排序
    era_order = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s+"]
    era_stats['era'] = pd.Categorical(era_stats['era'], categories=era_order, ordered=True)
    era_stats = era_stats.sort_values('era')
    
    plt.figure(figsize=(14, 8))
    plt.bar(era_stats['era'], era_stats['PPG(场均得分)'], color='royalblue')
    plt.title('NBA球员场均得分历史变化', fontsize=14)
    plt.xlabel('年代')
    plt.ylabel('场均得分(PPG)')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(era_stats['PPG(场均得分)']):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.savefig(os.path.join(output_dir, '8_得分历史变化.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 三分命中率历史变化
    plt.figure(figsize=(14, 8))
    plt.bar(era_stats['era'], era_stats['3P%(平均三分命中率)'], color='green')
    plt.title('NBA球员三分命中率历史变化', fontsize=14)
    plt.xlabel('年代')
    plt.ylabel('平均三分命中率')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(era_stats['3P%(平均三分命中率)']):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center')
    
    plt.savefig(os.path.join(output_dir, '9_三分历史变化.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"高级图表已保存到 {output_dir} 目录")

def create_interactive_charts(df, output_dir='output/visualizations'):
    """创建交互式图表：使用Plotly"""
    print("\n===== 创建交互式图表 =====")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 交互式散点图：身高-体重-效率值
    # 检查"效率值"列是否存在，如果不存在，看是否有"Efficiency"列
    if 'Efficiency' in df.columns and '效率值' not in df.columns:
        df['效率值'] = df['Efficiency']
        print("已将 'Efficiency' 列重命名为 '效率值'")
    
    # 首先过滤掉效率值为NaN的行
    df_filtered = df.dropna(subset=['效率值'] if '效率值' in df.columns else ['Efficiency'])
    print(f"原始数据行数: {len(df)}，过滤后数据行数: {len(df_filtered)}")
    
    # 确保身高和体重列存在
    height_col = '身高(英寸)' if '身高(英寸)' in df.columns else 'height_inches'
    weight_col = '体重(磅)' if '体重(磅)' in df.columns else 'weight_lbs'
    size_col = '效率值' if '效率值' in df.columns else 'Efficiency'
    
    fig = px.scatter(df_filtered, x=height_col, y=weight_col, 
                    color='position_group', size=size_col,
                    hover_name='name', size_max=50,
                    title='NBA球员身高-体重与效率值关系（交互式）')
    
    fig.update_layout(
        title_font_size=20,
        xaxis_title='身高(英寸)',
        yaxis_title='体重(磅)',
        legend_title='位置'
    )
    
    fig.write_html(os.path.join(output_dir, '10_交互式散点图.html'))
    
    # 2. 雷达图：不同位置球员能力对比
    # 确保使用正确的列名
    ppg_col = 'PPG(场均得分)' if 'PPG(场均得分)' in df.columns else 'PPG'
    rpg_col = 'RPG(场均篮板)' if 'RPG(场均篮板)' in df.columns else 'RPG'
    apg_col = 'APG(场均助攻)' if 'APG(场均助攻)' in df.columns else 'APG'
    spg_col = 'SPG(场均抢断)' if 'SPG(场均抢断)' in df.columns else 'SPG'
    bpg_col = 'BPG(场均盖帽)' if 'BPG(场均盖帽)' in df.columns else 'BPG'
    
    position_stats = df.groupby('position_group').agg({
        ppg_col: 'mean',
        rpg_col: 'mean',
        apg_col: 'mean',
        spg_col: 'mean',
        bpg_col: 'mean'
    }).reset_index()
    
    # 重命名列以便后续使用
    position_stats.rename(columns={
        ppg_col: 'PPG(场均得分)',
        rpg_col: 'RPG(场均篮板)',
        apg_col: 'APG(场均助攻)',
        spg_col: 'SPG(场均抢断)',
        bpg_col: 'BPG(场均盖帽)'
    }, inplace=True)
    
    categories = ['PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 'SPG(场均抢断)', 'BPG(场均盖帽)']
    
    fig = go.Figure()
    
    for i, pos in enumerate(position_stats['position_group']):
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
        title='不同位置NBA球员能力对比（交互式雷达图）'
    )
    
    fig.write_html(os.path.join(output_dir, '11_交互式雷达图.html'))
    
    # 3. 交互式折线图：历史趋势
    # 确保使用正确的列名
    ppg_col = 'PPG(场均得分)' if 'PPG(场均得分)' in df.columns else 'PPG'
    rpg_col = 'RPG(场均篮板)' if 'RPG(场均篮板)' in df.columns else 'RPG'
    apg_col = 'APG(场均助攻)' if 'APG(场均助攻)' in df.columns else 'APG'
    tp_pct_col = '3P%(平均三分命中率)' if '3P%(平均三分命中率)' in df.columns else 'Avg_3P_Pct'
    
    yearly_stats = df.groupby('year_start').agg({
        ppg_col: 'mean',
        rpg_col: 'mean',
        apg_col: 'mean',
        tp_pct_col: 'mean',
    }).reset_index()
    
    # 过滤掉样本太少的年份
    year_counts = df['year_start'].value_counts()
    valid_years = year_counts[year_counts > 5].index
    yearly_stats = yearly_stats[yearly_stats['year_start'].isin(valid_years)]
    
    # 重命名列以便后续使用
    yearly_stats.rename(columns={
        ppg_col: 'PPG(场均得分)',
        rpg_col: 'RPG(场均篮板)',
        apg_col: 'APG(场均助攻)',
        tp_pct_col: '3P%(平均三分命中率)'
    }, inplace=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly_stats['year_start'], y=yearly_stats['PPG(场均得分)'], mode='lines+markers', name='场均得分'))
    fig.add_trace(go.Scatter(x=yearly_stats['year_start'], y=yearly_stats['RPG(场均篮板)'], mode='lines+markers', name='场均篮板'))
    fig.add_trace(go.Scatter(x=yearly_stats['year_start'], y=yearly_stats['APG(场均助攻)'], mode='lines+markers', name='场均助攻'))
    fig.add_trace(go.Scatter(x=yearly_stats['year_start'], y=yearly_stats['3P%(平均三分命中率)'], mode='lines+markers', name='三分命中率'))
    
    fig.update_layout(
        title='NBA球员数据历史变化趋势（交互式）',
        xaxis_title='年份',
        yaxis_title='场均数据',
        legend_title='数据类型',
        hovermode="x unified"
    )
    
    fig.write_html(os.path.join(output_dir, '12_交互式历史趋势图.html'))
    
    # 4. 3D散点图
    # 修正：使用过滤掉NaN值的df_filtered，而不是原始的df
    fig = px.scatter_3d(df_filtered, x='PPG(场均得分)', y='RPG(场均篮板)', z='APG(场均助攻)',
                       color='position_group', size='效率值',
                       hover_name='name', opacity=0.7,
                       title='NBA球员得分-篮板-助攻三维分布')
    
    fig.update_layout(
        scene=dict(
            xaxis_title='场均得分',
            yaxis_title='场均篮板',
            zaxis_title='场均助攻'
        )
    )
    
    fig.write_html(os.path.join(output_dir, '14_3D散点图.html'))
    
    print(f"交互式图表已保存到 {output_dir} 目录")

def main():
    """主函数"""
    # 加载数据
    df = load_data()
    
    # 输出目录
    output_dir = 'output/visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建各种可视化图表
    create_basic_charts(df, output_dir)
    create_advanced_charts(df, output_dir)
    create_interactive_charts(df, output_dir)
    
    print(f"\n可视化分析完成。图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()
