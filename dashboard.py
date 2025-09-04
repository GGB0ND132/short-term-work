import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# --- 页面配置 ---
st.set_page_config(
    page_title="NBA球员数据分析",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 数据加载 (带缓存) ---
@st.cache_data
def load_data(path):
    """加载CSV数据"""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_html(path):
    """加载HTML文件内容"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

@st.cache_data
def load_text(path):
    """加载文本文件内容"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

# 加载所有需要的数据和结果
cleaned_df = load_data('cleaned_data.csv')
cluster_df = load_data('output/clustering/2_聚类分析结果.csv')
cluster_report = load_text('output/clustering/3_聚类分析报告.txt')
prediction_report = load_text('output/prediction/3_模型评估报告.txt')


# --- 侧边栏导航 ---
st.sidebar.title("🏀 分析导航")
page = st.sidebar.radio(
    "选择一个分析页面:",
    ["项目概览", "描述性统计", "数据可视化", "球员聚类分析", "得分预测分析"]
)

# --- 页面内容 ---

if page == "项目概览":
    st.title("NBA 球员数据分析与可视化平台")
    st.markdown("---")
    st.header("项目简介")
    st.write("""
    本项目旨在对NBA球员的历史数据进行深入分析。我们通过数据清洗、统计分析、可视化和机器学习等方法，探索球员表现的规律、对球员进行分类，并尝试预测球员的得分能力。
    - **数据来源**: [Kaggle NBA Players Stats](https://www.kaggle.com/datasets/drgilermo/nba-players-stats)
    - **技术栈**: Python, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn, Streamlit
    - **核心功能**:
        - 交互式数据探索
        - 多维度球员能力可视化
        - 基于K-Means的球员类型聚类
        - 基于机器学习的得分预测模型
    
    请使用左侧的导航栏切换不同的分析模块。
    """)
    
    st.header("数据集预览")
    if cleaned_df is not None:
        st.dataframe(cleaned_df.head(10))
    else:
        st.warning("未找到清洗后的数据 `cleaned_data.csv`。")

elif page == "描述性统计":
    st.title("描述性统计分析")
    st.markdown("---")
    
    st.header("核心指标统计")
    stats_path = 'output/1_整体统计指标.csv'
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path)
        st.dataframe(stats_df)
    else:
        st.warning("未找到核心指标统计文件。")

    st.header("按位置分组统计")
    position_stats_path = 'output/2_按位置分组统计.csv'
    if os.path.exists(position_stats_path):
        pos_stats_df = pd.read_csv(position_stats_path)
        st.dataframe(pos_stats_df)
    else:
        st.warning("未找到按位置分组统计文件。")
        
    st.header("联盟历史趋势")
    trends_path = 'output/4_年度趋势统计.csv'
    if os.path.exists(trends_path):
        trends_df = pd.read_csv(trends_path)
        st.dataframe(trends_df)
        
        st.subheader("联盟平均得分/三分/罚球趋势图")
        # 修正：将 x='Year' 改为 x='year_start'，并使用正确的Y轴列名
        fig = px.line(trends_df, x='year_start', y=['PPG(场均得分)', 'RPG(场均篮板)', 'APG(场均助攻)'],
                      title="联盟历史数据趋势", labels={'year_start': '年份'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("未找到联盟年度趋势文件。")


elif page == "数据可视化":
    st.title("数据可视化探索")
    st.markdown("---")
    
    st.header("静态图表")
    viz_dir = 'output/visualizations'
    if os.path.exists(viz_dir):
        col1, col2 = st.columns(2)
        with col1:
            st.image(os.path.join(viz_dir, '1_身高分布直方图.png'), caption='球员身高分布')
            st.image(os.path.join(viz_dir, '3_身高体重散点图.png'), caption='身高与体重关系')
        with col2:
            st.image(os.path.join(viz_dir, '2_位置得分箱线图.png'), caption='不同位置得分')
            st.image(os.path.join(viz_dir, '6_相关性热力图.png'), caption='核心数据相关性')
    else:
        st.warning("未找到可视化图片目录。")

    st.header("交互式图表")
    interactive_path = os.path.join(viz_dir, '10_交互式散点图.html')
    if os.path.exists(interactive_path):
        st.subheader("身高-体重-效率值关系")
        html_content = load_html(interactive_path)
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.warning("未找到交互式散点图文件。")


elif page == "球员聚类分析":
    st.title("K-Means 球员聚类分析")
    st.markdown("---")
    
    st.header("聚类分析报告")
    if cluster_report:
        st.text(cluster_report)
    else:
        st.warning("未找到聚类分析报告。")

    st.header("聚类结果概览")
    if cluster_df is not None:
        st.dataframe(cluster_df)
    else:
        st.warning("未找到聚类结果文件。")
        
    st.header("交互式聚类可视化")
    cluster_viz_path = 'output/clustering/5_交互式聚类可视化.html'
    if os.path.exists(cluster_viz_path):
        html_content = load_html(cluster_viz_path)
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.warning("未找到交互式聚类可视化文件。")
        
    st.header("球员类型能力雷达图")
    radar_path = 'output/clustering/6_聚类雷达图.html'
    if os.path.exists(radar_path):
        html_content = load_html(radar_path)
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.warning("未找到聚类雷达图文件。")


elif page == "得分预测分析":
    st.title("球员得分预测分析")
    st.markdown("---")
    
    st.header("模型评估报告")
    if prediction_report:
        st.text(prediction_report)
    else:
        st.warning("未找到模型评估报告。")
        
    st.header("预测结果可视化")
    pred_dir = 'output/prediction'
    if os.path.exists(pred_dir):
        st.image(os.path.join(pred_dir, '1_预测vs实际.png'), caption='随机森林模型预测结果')
        st.image(os.path.join(pred_dir, '4_特征重要性.png'), caption='模型特征重要性排序')
    else:
        st.warning("未找到预测结果图片。")

st.sidebar.info("平台由 Streamlit 构建 | 2025年")