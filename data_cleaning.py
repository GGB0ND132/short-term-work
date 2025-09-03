
"""
NBA球员数据清洗脚本
- 处理缺失值和异常值
- 创建有用的新特征
- 保存清洗后的数据
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置随机种子，保证结果可重现
np.random.seed(42)

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 定义数据文件路径
DATA_PATH = 'raw-data'
PLAYERS_PATH = os.path.join(DATA_PATH, 'Players.csv')
SEASONS_STATS_PATH = os.path.join(DATA_PATH, 'Seasons_Stats.csv')
PLAYER_DATA_PATH = os.path.join(DATA_PATH, 'player_data.csv')
OUTPUT_PATH = 'cleaned_data.csv'

def load_data():
    """
    加载所有数据文件
    """
    print("正在加载数据...")
    players_df = pd.read_csv(PLAYERS_PATH)
    seasons_stats_df = pd.read_csv(SEASONS_STATS_PATH)
    player_data_df = pd.read_csv(PLAYER_DATA_PATH)
    
    print(f"Players 数据形状: {players_df.shape}")
    print(f"Seasons_Stats 数据形状: {seasons_stats_df.shape}")
    print(f"player_data 数据形状: {player_data_df.shape}")
    
    return players_df, seasons_stats_df, player_data_df

def handle_missing_values(df, strategy='median'):
    """
    处理缺失值
    
    参数:
        df: 数据框
        strategy: 处理数值型缺失值的策略，可选 'median', 'mean', 'drop'
    
    返回:
        处理后的数据框
    """
    print(f"\n处理缺失值前的形状: {df.shape}")
    
    # 显示缺失值情况
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    missing_percent = (missing_values / len(df)) * 100
    
    if not missing_values.empty:
        print("\n缺失值情况:")
        for col, count in missing_values.items():
            print(f"{col}: {count} 缺失值 ({missing_percent[col]:.2f}%)")
    
    # 根据数据类型分别处理缺失值
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # 处理数值型缺失值
            if np.issubdtype(df[col].dtype, np.number):
                if strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'drop' and missing_percent[col] < 30:  # 如果缺失比例不高，可以删除
                    df = df.dropna(subset=[col])
            # 处理分类型和字符串型缺失值
            elif df[col].dtype == 'object':
                # 对于字符串型，如果缺失值过多（>50%），考虑删除该列
                if missing_percent[col] > 50:
                    print(f"列 {col} 缺失值过多 ({missing_percent[col]:.2f}%)，考虑删除")
                    # 此处不直接删除，而是将其标记，供手动决定
                else:
                    # 用"Unknown"填充
                    df[col] = df[col].fillna("Unknown")
    
    print(f"处理缺失值后的形状: {df.shape}")
    return df

def handle_outliers(df, num_columns=None, method='iqr', threshold=3):
    """
    处理异常值
    
    参数:
        df: 数据框
        num_columns: 需要处理异常值的数值型列名列表，默认为None表示处理所有数值型列
        method: 处理方法，可选 'iqr' (四分位距法) 或 'zscore' (z-score法)
        threshold: 阈值，对于z-score法是标准差的倍数，对于iqr法是四分位距的倍数
    
    返回:
        处理后的数据框
    """
    print("\n处理异常值...")
    df_cleaned = df.copy()
    
    if num_columns is None:
        num_columns = df.select_dtypes(include=np.number).columns
    
    for col in num_columns:
        if method == 'iqr':
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # 计算异常值的数量
            outliers_count = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
            outlier_percent = (outliers_count / len(df_cleaned)) * 100
            
            if outliers_count > 0:
                print(f"列 {col} 有 {outliers_count} 个异常值 ({outlier_percent:.2f}%)")
                
                # 将异常值替换为边界值
                df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                
        elif method == 'zscore':
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            
            # 计算异常值的数量
            outliers_count = ((df_cleaned[col] - mean).abs() > threshold * std).sum()
            outlier_percent = (outliers_count / len(df_cleaned)) * 100
            
            if outliers_count > 0:
                print(f"列 {col} 有 {outliers_count} 个异常值 ({outlier_percent:.2f}%)")
                
                # 将异常值替换为边界值
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
    
    return df_cleaned

def extract_height_weight_from_player_data(player_data_df):
    """
    从player_data中提取身高和体重数据（需要转换格式）
    """
    print("\n正在处理身高体重数据...")
    
    # 备份原始列
    player_data_df['height_original'] = player_data_df['height']
    player_data_df['weight_original'] = player_data_df['weight']
    
    # 提取身高（从"6-10"格式转换为英寸）
    def convert_height_to_inches(height_str):
        if pd.isna(height_str):
            return np.nan
        try:
            parts = height_str.split('-')
            if len(parts) == 2:
                feet = int(parts[0])
                inches = int(parts[1])
                return feet * 12 + inches
        except:
            return np.nan
        return np.nan
    
    player_data_df['height_inches'] = player_data_df['height'].apply(convert_height_to_inches)
    
    # 提取体重（去除单位）
    def convert_weight(weight_str):
        if pd.isna(weight_str):
            return np.nan
        try:
            return float(str(weight_str))
        except:
            return np.nan
    
    player_data_df['weight_lbs'] = player_data_df['weight'].apply(convert_weight)
    
    return player_data_df

def extract_date_from_birth_date(player_data_df):
    """
    从birth_date列提取生日信息
    """
    print("\n正在处理出生日期数据...")
    
    def parse_birth_date(date_str):
        if pd.isna(date_str):
            return np.nan
        try:
            # 处理多种可能的日期格式
            formats = [
                "%B %d, %Y",    # April 16, 1947
                "%b %d, %Y",    # Apr 16, 1947
                "%d-%b-%Y",     # 16-Apr-1947
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except:
                    continue
                    
            return np.nan
        except:
            return np.nan
    
    # 解析出生日期
    player_data_df['birth_date_parsed'] = player_data_df['birth_date'].apply(parse_birth_date)
    
    # 提取年、月、日
    player_data_df['birth_year'] = player_data_df['birth_date_parsed'].dt.year
    player_data_df['birth_month'] = player_data_df['birth_date_parsed'].dt.month
    player_data_df['birth_day'] = player_data_df['birth_date_parsed'].dt.day
    
    return player_data_df

def create_features(players_df, seasons_stats_df, player_data_df):
    """
    创建新特征
    """
    print("\n创建新特征...")
    
    # 1. 合并球员基本信息数据
    # 首先处理player_data_df中的特殊数据格式
    player_data_df = extract_height_weight_from_player_data(player_data_df)
    player_data_df = extract_date_from_birth_date(player_data_df)
    
    # 2. 计算生涯年份
    player_data_df['career_length'] = player_data_df['year_end'] - player_data_df['year_start'] + 1
    
    # 3. 从赛季数据计算生涯统计数据
    # 按球员分组，计算生涯数据
    career_stats = seasons_stats_df.groupby('Player').agg({
        'G': 'sum',           # 总比赛场次
        'PTS': 'sum',         # 总得分
        'TRB': 'sum',         # 总篮板
        'AST': 'sum',         # 总助攻
        'STL': 'sum',         # 总抢断
        'BLK': 'sum',         # 总盖帽
        'Year': 'count',      # 赛季数量
        'PER': 'mean',        # 平均效率值
        'WS': 'sum',          # 总胜利贡献值
        'FG%': 'mean',        # 平均投篮命中率
        '3P%': 'mean',        # 平均三分命中率
        'FT%': 'mean',        # 平均罚球命中率
    }).reset_index()
    
    # 重命名列
    career_stats.columns = ['Player', 'Games_Played', 'Total_Points', 'Total_Rebounds', 'Total_Assists', 
                            'Total_Steals', 'Total_Blocks', 'Seasons_Count', 'Avg_PER', 'Total_Win_Shares',
                            'Avg_FG_Pct', 'Avg_3P_Pct', 'Avg_FT_Pct']
    
    # 4. 计算生涯场均数据
    career_stats['PPG'] = career_stats['Total_Points'] / career_stats['Games_Played']
    career_stats['RPG'] = career_stats['Total_Rebounds'] / career_stats['Games_Played']
    career_stats['APG'] = career_stats['Total_Assists'] / career_stats['Games_Played']
    career_stats['SPG'] = career_stats['Total_Steals'] / career_stats['Games_Played']
    career_stats['BPG'] = career_stats['Total_Blocks'] / career_stats['Games_Played']
    
    # 5. 创建"效率值"特征
    # 使用公式: (PTS + RPG + APG + SPG + BPG) / 5
    career_stats['Efficiency'] = (
        career_stats['PPG'] + career_stats['RPG'] + career_stats['APG'] + 
        career_stats['SPG'] + career_stats['BPG']
    ) / 5
    
    # 6. 判断是否为全明星球员（基于效率值和胜利贡献值）
    # 这里使用简单的效率值阈值作为示例，实际全明星认定需要更复杂的逻辑
    career_stats['Is_AllStar'] = 0  # 默认不是全明星
    
    # 将效率值排名前15%的球员视为全明星
    allstar_threshold = career_stats['Efficiency'].quantile(0.85)
    career_stats.loc[career_stats['Efficiency'] >= allstar_threshold, 'Is_AllStar'] = 1
    
    # 将胜利贡献值排名前15%的球员也视为全明星
    ws_threshold = career_stats['Total_Win_Shares'].quantile(0.85)
    career_stats.loc[career_stats['Total_Win_Shares'] >= ws_threshold, 'Is_AllStar'] = 1
    
    # 7. 合并数据
    # 将career_stats与player_data_df合并
    # 使用name列与Player列进行合并
    merged_df = pd.merge(player_data_df, career_stats, left_on='name', right_on='Player', how='left')
    
    # 删除重复的列
    if 'Player' in merged_df.columns:
        merged_df = merged_df.drop(columns=['Player'])
    
    return merged_df

def main():
    """
    主函数：加载数据、清洗数据、创建特征、保存结果
    """
    # 加载数据
    players_df, seasons_stats_df, player_data_df = load_data()
    
    # 处理缺失值
    players_df = handle_missing_values(players_df)
    seasons_stats_df = handle_missing_values(seasons_stats_df)
    player_data_df = handle_missing_values(player_data_df)
    
    # 处理异常值（主要针对数值型特征）
    seasons_stats_df = handle_outliers(
        seasons_stats_df, 
        num_columns=['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'PER', 'WS'],
        method='iqr'
    )
    
    # 创建新特征
    cleaned_data = create_features(players_df, seasons_stats_df, player_data_df)
    
    # 打印清洗后的数据集信息
    print("\n清洗后的数据集信息:")
    print(f"形状: {cleaned_data.shape}")
    print(f"列: {cleaned_data.columns.tolist()}")
    print(f"前5行:\n{cleaned_data.head()}")
    
    # 保存清洗后的数据
    cleaned_data.to_csv(OUTPUT_PATH, index=False)
    print(f"\n清洗后的数据已保存到: {OUTPUT_PATH}")
    
    return cleaned_data

if __name__ == "__main__":
    main()
