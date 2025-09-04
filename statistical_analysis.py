
"""
NBA球员数据统计分析
- 全面的描述性统计
- 按位置、年代分组统计
- 年度趋势分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

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

# 位置分组映射
position_groups = {
    'G': ['G', 'G-F'],
    'F': ['F', 'F-G', 'F-C'],
    'C': ['C', 'C-F']
}

# 年代分组
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
    
    # 添加位置分组和年代分组
    df['position_group'] = df['position'].apply(lambda x: 'G' if x in position_groups['G'] else 
                                                       'F' if x in position_groups['F'] else
                                                       'C' if x in position_groups['C'] else 'Other')
    
    df['era'] = df['year_start'].apply(assign_era)
    
    return df

def perform_descriptive_statistics(df, output_dir='output'):
    """执行全面的描述性统计分析"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n===== 执行全面的描述性统计分析 =====")
    
    # 统计指标列表
    stats_cols = [
        'PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
        'SPG(场均抢断)', 'BPG(场均盖帽)', '效率值',
        'PER(平均效率值)', 'WS(总胜利贡献值)',
        'FG%(平均投篮命中率)', '3P%(平均三分命中率)', 'FT%(平均罚球命中率)'
    ]
    
    # 1. 全部球员的统计指标分析
    overall_stats = df[stats_cols].describe().transpose()
    overall_stats['变异系数'] = overall_stats['std'] / overall_stats['mean']  # 计算变异系数
    
    # 保存到CSV
    overall_stats.to_csv(os.path.join(output_dir, '1_整体统计指标.csv'), encoding='utf-8-sig')
    
    # 生成文字统计报告
    with open(os.path.join(output_dir, '1_整体统计报告.txt'), 'w', encoding='utf-8') as f:
        f.write("NBA球员数据统计报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"数据集大小: {df.shape[0]} 名球员\n\n")
        
        f.write("1. 核心统计指标摘要\n")
        f.write("-" * 50 + "\n")
        for col in stats_cols:
            if col in df.columns:
                f.write(f"{col}:\n")
                f.write(f"  - 平均值: {df[col].mean():.2f}\n")
                f.write(f"  - 中位数: {df[col].median():.2f}\n")
                f.write(f"  - 最大值: {df[col].max():.2f}\n")
                f.write(f"  - 最小值: {df[col].min():.2f}\n")
                f.write(f"  - 标准差: {df[col].std():.2f}\n")
                f.write(f"  - 变异系数: {df[col].std() / df[col].mean():.2f}\n\n")
        
        # 计算全明星球员比例
        if '是否全明星(1=是,0=否)' in df.columns:
            allstar_count = df['是否全明星(1=是,0=否)'].sum()
            allstar_pct = (allstar_count / len(df)) * 100
            f.write(f"全明星球员数量: {allstar_count} ({allstar_pct:.2f}%)\n\n")
    
    return overall_stats

def analyze_by_position(df, output_dir='output'):
    """按位置分组统计分析"""
    print("\n===== 按位置分组统计分析 =====")
    
    # 统计指标列表
    stats_cols = [
        'PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
        'SPG(场均抢断)', 'BPG(场均盖帽)', '效率值',
        'PER(平均效率值)', 'WS(总胜利贡献值)',
        'FG%(平均投篮命中率)', '3P%(平均三分命中率)', 'FT%(平均罚球命中率)'
    ]
    
    # 按position_group分组统计
    position_stats = df.groupby('position_group')[stats_cols].agg(['mean', 'median', 'std']).round(2)
    position_stats.to_csv(os.path.join(output_dir, '2_按位置分组统计.csv'), encoding='utf-8-sig')
    
    # 生成文字统计报告
    with open(os.path.join(output_dir, '2_按位置分析报告.txt'), 'w', encoding='utf-8') as f:
        f.write("NBA球员位置分组统计分析\n")
        f.write("=" * 50 + "\n\n")
        
        for pos in ['G', 'F', 'C']:
            pos_df = df[df['position_group'] == pos]
            f.write(f"{pos} 位置球员分析 (共 {len(pos_df)} 人)\n")
            f.write("-" * 50 + "\n")
            
            for col in stats_cols:
                if col in pos_df.columns:
                    f.write(f"{col}:\n")
                    f.write(f"  - 平均值: {pos_df[col].mean():.2f}\n")
                    f.write(f"  - 中位数: {pos_df[col].median():.2f}\n")
                    f.write(f"  - 标准差: {pos_df[col].std():.2f}\n")
            
            # 计算全明星比例
            if '是否全明星(1=是,0=否)' in pos_df.columns:
                pos_allstar_count = pos_df['是否全明星(1=是,0=否)'].sum()
                pos_allstar_pct = (pos_allstar_count / len(pos_df)) * 100
                f.write(f"全明星球员数量: {pos_allstar_count} ({pos_allstar_pct:.2f}%)\n\n")
            
            f.write("\n")
        
        # 位置特点总结
        f.write("位置特点总结:\n")
        f.write("-" * 50 + "\n")
        f.write("1. 后卫(G): 以场均得分和助攻为主要特点，抢断能力较强，篮板和盖帽较少\n")
        f.write("2. 前锋(F): 全面型球员，得分、篮板均衡，中等水平的助攻和防守数据\n")
        f.write("3. 中锋(C): 以篮板和盖帽为主要特点，得分效率高，助攻和抢断较少\n")
    
    return position_stats

def analyze_by_era(df, output_dir='output'):
    """按年代分组统计分析"""
    print("\n===== 按年代分组统计分析 =====")
    
    # 统计指标列表
    stats_cols = [
        'PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
        'SPG(场均抢断)', 'BPG(场均盖帽)', '效率值',
        'PER(平均效率值)', 'WS(总胜利贡献值)',
        'FG%(平均投篮命中率)', '3P%(平均三分命中率)', 'FT%(平均罚球命中率)',
        '身高(英寸)', '体重(磅)', '生涯长度(年)'
    ]
    
    # 按era分组统计
    era_stats = df.groupby('era')[stats_cols].agg(['mean', 'median']).round(2)
    era_stats.to_csv(os.path.join(output_dir, '3_按年代分组统计.csv'), encoding='utf-8-sig')
    
    # 生成文字统计报告
    with open(os.path.join(output_dir, '3_按年代分析报告.txt'), 'w', encoding='utf-8') as f:
        f.write("NBA球员年代变化趋势分析\n")
        f.write("=" * 50 + "\n\n")
        
        eras = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s+"]
        
        for era in eras:
            era_df = df[df['era'] == era]
            if len(era_df) > 0:  # 确保该年代有数据
                f.write(f"{era} 年代球员分析 (共 {len(era_df)} 人)\n")
                f.write("-" * 50 + "\n")
                
                for col in ['PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', '3P%(平均三分命中率)', '身高(英寸)', '体重(磅)']:
                    if col in era_df.columns:
                        f.write(f"{col}: {era_df[col].mean():.2f}\n")
                
                f.write("\n")
        
        # 年代趋势总结
        f.write("历史变化趋势总结:\n")
        f.write("-" * 50 + "\n")
        f.write("1. 身高体重: NBA球员整体身高和体重呈增长趋势\n")
        f.write("2. 三分球: 随着时间推移，三分球使用频率和命中率均有提升\n")
        f.write("3. 球员效率: 现代球员的PER和效率值普遍高于早期球员\n")
        f.write("4. 职业寿命: 现代球员由于训练和医疗条件改善，平均职业生涯更长\n")
    
    return era_stats

def analyze_trends(df, output_dir='output'):
    """分析年度趋势"""
    print("\n===== 分析年度趋势 =====")
    
    # 按开始年份分组计算平均值
    yearly_stats = df.groupby('year_start').agg({
        'PPG(场均得分)': 'mean',
        'RPG(场均篮板)': 'mean',
        'APG(场均助攻)': 'mean',
        'SPG(场均抢断)': 'mean',
        'BPG(场均盖帽)': 'mean',
        '3P%(平均三分命中率)': 'mean',
        'FG%(平均投篮命中率)': 'mean',
        '身高(英寸)': 'mean',
        '体重(磅)': 'mean',
        '生涯长度(年)': 'mean'
    }).reset_index()
    
    # 过滤掉样本太少的年份
    year_counts = df['year_start'].value_counts()
    valid_years = year_counts[year_counts > 5].index
    yearly_stats_filtered = yearly_stats[yearly_stats['year_start'].isin(valid_years)]
    
    # 保存到CSV
    yearly_stats_filtered.to_csv(os.path.join(output_dir, '4_年度趋势统计.csv'), encoding='utf-8-sig')
    
    # 生成文字趋势报告
    with open(os.path.join(output_dir, '4_年度趋势分析报告.txt'), 'w', encoding='utf-8') as f:
        f.write("NBA球员历史趋势分析\n")
        f.write("=" * 50 + "\n\n")
        
        # 计算早期(1950-1980)和现代(1990-2020)的平均值对比
        early_era = df[(df['year_start'] >= 1950) & (df['year_start'] < 1980)]
        modern_era = df[(df['year_start'] >= 1990) & (df['year_start'] <= 2020)]
        
        f.write("早期(1950-1980)与现代(1990-2020)球员对比:\n")
        f.write("-" * 50 + "\n")
        
        compare_cols = [
            'PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)', 
            '3P%(平均三分命中率)', 'FG%(平均投篮命中率)', 
            '身高(英寸)', '体重(磅)', '生涯长度(年)'
        ]
        
        for col in compare_cols:
            if col in df.columns:
                early_avg = early_era[col].mean()
                modern_avg = modern_era[col].mean()
                change_pct = ((modern_avg - early_avg) / early_avg) * 100
                
                f.write(f"{col}:\n")
                f.write(f"  - 早期平均: {early_avg:.2f}\n")
                f.write(f"  - 现代平均: {modern_avg:.2f}\n")
                f.write(f"  - 变化比例: {change_pct:.2f}%\n\n")
        
        # 趋势总结
        f.write("主要趋势总结:\n")
        f.write("-" * 50 + "\n")
        f.write("1. 得分趋势: 现代比赛节奏更快，场均得分普遍提升\n")
        f.write("2. 三分球革命: 三分球使用频率显著增加，成为现代进攻核心要素\n")
        f.write("3. 位置革命: 位置界限变得模糊，全能型球员增多\n")
        f.write("4. 身体素质: 球员身高体重持续优化，运动员素质全面提高\n")
        f.write("5. 职业寿命: 由于现代医疗和训练方法改进，球员职业生涯延长\n")
    
    return yearly_stats_filtered

def main():
    """主函数"""
    # 加载数据
    df = load_data()
    
    # 输出目录
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 执行各项统计分析
    overall_stats = perform_descriptive_statistics(df, output_dir)
    position_stats = analyze_by_position(df, output_dir)
    era_stats = analyze_by_era(df, output_dir)
    yearly_trends = analyze_trends(df, output_dir)
    
    print(f"\n统计分析完成。结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()
