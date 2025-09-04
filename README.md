这是一个homework

# ！无法直连github 替换为镜像源
```powershell
# 原始地址
# git clone https://github.com/GGB0ND132/short-term-work.git

# 使用镜像地址（任选一个）
git clone https://hub.nuaa.cf/GGB0ND132/short-term-work.git
git clone https://hub.yzuu.cf/GGB0ND132/short-term-work.git
git clone https://gitclone.com/github.com/GGB0ND132/short-term-work.git
```

# 一、环境配置
1. 安装uv
```powershell
pip install uv
```

2. 检查uv环境
```powershell
uv --version
```
如果有报错，请移步 https://github.com/astral-sh/uv 看看教程

3. 创建虚拟环境并安装依赖
```powershell
cd your-local-project-path
uv sync
```
4. 激活虚拟环境 (可选)
```powershell
# 激活虚拟环境，这样你的所有Python操作都会在这个隔离环境中进行
.\.venv\Scripts\activate
```

# 二、操作指南
1.	Fork 本项目。点击 Fork 按钮，创建一个新的派生项目到自己的工作区（Create a new fork）

2.	克隆派生
```powershell
# 克隆项目到本地（注意是派生项目的链接，不是原始项目）
git clone https://github.com/your-user-name/your-fork-name.git
```

3.	创建分支
```powershell
# 创建并切换到本地新分支，分支的命名尽量简洁，并与想要解决的问题相关
git checkout -b your-branch-name
```

4.	修改代码内容或者新增代码删除代码等

5.	提交更改 
```powershell
git commit -m 'your-commit-content'
```

6.	推送到分支
```powershell
git push --set-upstream origin your-branch-name
```
7.	提交合并请求
提交时添加标题和描述信息
8. 等待审核，审核通过后，合并分支到主分支

9. 如果在你提交之前，原始项目有更新，请同步更新到你的派生项目
``` powershell
git remote add upstream https://github.com/GGB0ND132/short-term-work.git
git fetch upstream
git checkout your-branch-name
git merge upstream/master
cd your-local-project-path
uv sync
```

# 任务安排

## 一、每个人都要仔细阅读并理解项目要求。

一起查看Kaggle数据集，了解各个字段的含义（如PTS, TRB, AST, FG%等）。

讨论最终想分析什么主题？（例如：联盟得分趋势、巨星技术特点、球员类型聚类、薪资与表现关系等）

统一开发环境：确保每个人都成功配置好uv和虚拟环境（uv sync）。

## 二、分工与执行 (4-5天)

1. 数据清洗 
- 主导任务：

任务规划： 制定详细的时间节点，主持每日站会，同步进度，协调阻塞问题。

数据获取与清洗： 负责从Kaggle下载数据集，编写数据清洗的代码。

数据整合： 为团队提供清洗后的干净数据文件。

- 具体工作：

处理缺失值、异常值。

创建有用的新特征（如“生涯年份”、“是否全明星”、“效率值”等）。

将清洗后的数据保存为cleaned_data.csv供团队使用。

- 产出物： 清洗脚本(data_cleaning.py)，干净的数据集。

2. 描述性统计
- 主导任务： 

对清洗后的数据进行全面的描述性统计，发现数据的基本规律。

- 具体工作：

计算核心指标（得分、篮板、助攻等）的平均值、中位数、标准差、最大/最小值。

按位置（G, F, C）、年代进行分组统计，对比差异。

分析年度趋势（如联盟平均得分如何随时间变化）。

- 产出物： 统计结果，包括图表和文字结论。

3. 可视化 
- 主导任务： 将统计分析的结果用各种图表清晰、美观地呈现出来。

- 具体工作：

- 基础图表： 绘制直方图、箱线图、散点图、折线图（使用Matplotlib/Seaborn）。

- 高级图表： 绘制雷达图（用于球员对比）、热力图（用于相关性分析）、条形图（用于排名）。

- 交互式图表： 使用Plotly制作可交互的图表，丰富报告形式。

- 产出物： 所有可视化图表（保存为.png或.html文件），可视化脚本(visualization.py)。

4. 数据挖掘工
- 主导任务： 进行更深入的挖掘分析，运用机器学习算法。

- 具体工作：

球员聚类： 使用K-Means等算法，根据技术统计将球员自动分成几类（如“组织核心”、“3D球员”、“内线大闸”），并解读每类球员的特征。

回归预测： 尝试建立一个模型，根据球员的多项数据（如上场时间、出手次数）预测其得分。

- （可选）相关性分析： 分析各项数据之间的相关性，例如助攻数与胜利贡献值的关系。

- 产出物： 数据挖掘脚本(clustering.py, prediction.py)，模型结果，聚类图表。

5. 报告撰写与整合
- 主导任务： 负责最终调研报告的撰写、所有结果的整合与排版。

- 具体工作：

- 编写报告： 根据大纲，整合其他同学的分析结果、图表和结论，撰写国内外发展历程、实践探索等章节。

- 版本控制： 负责Git操作，管理分支，解决合并冲突，最终提交PR。

- 最终整合： 确保代码结构清晰，注释完整，README文件编写规范。

- 格式审查： 确保报告格式符合要求，参考文献引用规范。

- 产出物： 最终的Word报告文档、整洁的代码仓库。

