这是一个homework

# 一、贡献指南
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
git remote add upstream https://github.com/645359132/short-term1-homework.git
git fetch upstream
git checkout your-branch-name
git merge upstream/main
cd your-local-project-path
uv sync
```
# 二、环境配置
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