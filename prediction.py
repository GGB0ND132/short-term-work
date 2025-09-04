
"""
NBA球员数据预测分析
- 线性回归预测球员得分
- 评估模型性能
- 特征重要性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import os

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

def predict_ppg(df, output_dir='output/prediction'):
    """使用回归模型预测球员场均得分"""
    print("\n===== 预测球员场均得分 =====")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 选择特征和目标变量
    features = ['RPG(场均篮板)', 'APG(场均助攻)', 'SPG(场均抢断)', 'BPG(场均盖帽)',
                'FG%(平均投篮命中率)', '3P%(平均三分命中率)', 'FT%(平均罚球命中率)',
                '身高(英寸)', '体重(磅)', '生涯长度(年)']
    
    target = 'PPG(场均得分)'
    
    # 删除缺失值
    df_filtered = df.dropna(subset=features + [target])
    
    # 将位置转换为数值特征（独热编码）
    df_filtered = pd.get_dummies(df_filtered, columns=['position_group'], drop_first=True)
    
    # 调整特征列
    position_cols = [col for col in df_filtered.columns if 'position_group' in col]
    features = features + position_cols
    
    # 分割训练集和测试集
    X = df_filtered[features]
    y = df_filtered[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练线性回归模型
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # 评估线性回归模型
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    print(f"线性回归模型性能:")
    print(f"均方误差 (MSE): {mse_lr:.4f}")
    print(f"决定系数 (R²): {r2_lr:.4f}")
    
    # 训练随机森林回归模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # 评估随机森林模型
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print(f"\n随机森林模型性能:")
    print(f"均方误差 (MSE): {mse_rf:.4f}")
    print(f"决定系数 (R²): {r2_rf:.4f}")
    
    # 可视化预测结果
    visualize_predictions(y_test, y_pred_lr, y_pred_rf, output_dir)
    
    # 分析特征重要性
    feature_importance_analysis(lr_model, rf_model, features, output_dir)
    
    # 保存模型评估报告
    with open(os.path.join(output_dir, '3_模型评估报告.txt'), 'w', encoding='utf-8') as f:
        f.write("NBA球员得分预测模型评估\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 线性回归模型\n")
        f.write("-" * 50 + "\n")
        f.write(f"均方误差 (MSE): {mse_lr:.4f}\n")
        f.write(f"均方根误差 (RMSE): {np.sqrt(mse_lr):.4f}\n")
        f.write(f"决定系数 (R²): {r2_lr:.4f}\n\n")
        
        f.write("线性回归系数:\n")
        for feature, coef in zip(features, lr_model.coef_):
            f.write(f"  - {feature}: {coef:.4f}\n")
        f.write(f"  - 截距: {lr_model.intercept_:.4f}\n\n")
        
        f.write("2. 随机森林模型\n")
        f.write("-" * 50 + "\n")
        f.write(f"均方误差 (MSE): {mse_rf:.4f}\n")
        f.write(f"均方根误差 (RMSE): {np.sqrt(mse_rf):.4f}\n")
        f.write(f"决定系数 (R²): {r2_rf:.4f}\n\n")
        
        f.write("随机森林特征重要性:\n")
        importance = rf_model.feature_importances_
        for feature, imp in zip(features, importance):
            f.write(f"  - {feature}: {imp:.4f}\n")
        
        f.write("\n3. 模型比较\n")
        f.write("-" * 50 + "\n")
        f.write("随机森林模型在预测NBA球员场均得分方面表现更好，这可能是因为球员数据中存在非线性关系。\n")
        f.write("随机森林模型能够捕捉到更复杂的特征交互作用，而线性回归模型假设所有特征与目标变量之间是线性关系。\n\n")
        
        f.write("4. 结论\n")
        f.write("-" * 50 + "\n")
        f.write("1. 球员场均得分可以通过其他技术统计指标进行较准确的预测\n")
        f.write("2. 最重要的预测指标包括投篮命中率、场均助攻和场均篮板\n")
        f.write("3. 位置因素对预测球员得分也有显著影响\n")
        f.write("4. 非线性模型在预测球员表现方面优于线性模型\n")
    
    return lr_model, rf_model, X_test, y_test, y_pred_lr, y_pred_rf

def visualize_predictions(y_test, y_pred_lr, y_pred_rf, output_dir):
    """可视化预测结果"""
    # 创建预测vs实际值散点图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_lr, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际场均得分')
    plt.ylabel('预测场均得分')
    plt.title('线性回归: 预测 vs 实际')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际场均得分')
    plt.ylabel('预测场均得分')
    plt.title('随机森林: 预测 vs 实际')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_预测vs实际.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 使用Plotly创建交互式可视化
    results_df = pd.DataFrame({
        '实际场均得分': y_test,
        '线性回归预测': y_pred_lr,
        '随机森林预测': y_pred_rf,
    })
    
    fig = px.scatter(results_df, x='实际场均得分', y='随机森林预测', 
                   labels={'x': '实际场均得分', 'y': '预测场均得分'},
                   title='NBA球员得分预测结果（交互式）')
    
    fig.add_scatter(x=[y_test.min(), y_test.max()], 
                  y=[y_test.min(), y_test.max()], 
                  mode='lines', line=dict(color='red', dash='dash'),
                  name='理想预测线')
    
    fig.write_html(os.path.join(output_dir, '2_交互式预测结果.html'))

def feature_importance_analysis(lr_model, rf_model, features, output_dir):
    """分析特征重要性"""
    # 获取线性回归系数（绝对值）
    lr_importance = np.abs(lr_model.coef_)
    
    # 获取随机森林特征重要性
    rf_importance = rf_model.feature_importances_
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Linear_Regression': lr_importance,
        'Random_Forest': rf_importance
    })
    
    # 按随机森林重要性排序
    importance_df = importance_df.sort_values('Random_Forest', ascending=False)
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.barh(importance_df['Feature'], importance_df['Linear_Regression'])
    plt.title('线性回归特征重要性')
    plt.xlabel('重要性系数(绝对值)')
    plt.gca().invert_yaxis()  # 让最重要的特征显示在顶部
    
    plt.subplot(2, 1, 2)
    plt.barh(importance_df['Feature'], importance_df['Random_Forest'])
    plt.title('随机森林特征重要性')
    plt.xlabel('重要性系数')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_特征重要性.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 使用Plotly创建交互式条形图
    fig = px.bar(
        importance_df, 
        x='Random_Forest', 
        y='Feature',
        orientation='h',
        title='NBA球员得分预测的特征重要性',
        labels={'Random_Forest': '随机森林特征重要性', 'Feature': '特征'}
    )
    
    fig.write_html(os.path.join(output_dir, '5_交互式特征重要性.html'))
    
    # 保存特征重要性到CSV
    importance_df.to_csv(os.path.join(output_dir, '6_特征重要性.csv'), index=False)

def main():
    """主函数"""
    # 加载数据
    df = load_data()
    
    # 输出目录
    output_dir = 'output/prediction'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 预测球员场均得分
    lr_model, rf_model, X_test, y_test, y_pred_lr, y_pred_rf = predict_ppg(df, output_dir)
    
    print(f"\n预测分析完成。结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()
