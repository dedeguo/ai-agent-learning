# AI Basic

学习路线：最小数学与神经网络 → Attention → Transformer → GPT → Agent 与评测。

本项目用于学习大模型与智能体原理，并通过 Python 练习、Notebook 和小型项目巩固理解。基础阶段完成后，结合 CMU 11-768 的视频与 Stanford CS329Z 的作业，构建可运行、可评测的智能体应用。

## 学习路线与文档入口

从[总体学习计划](docs/学习计划.md)了解基础路线与验收标准。默认每周投入 6～8 小时，按掌握程度推进；周次表示学习顺序，不表示已完成进度。

| 阶段 | 内容 | 学习入口 |
| --- | --- | --- |
| 第 1 周 | 向量与矩阵、softmax、损失与梯度下降 | [最小数学与神经网络](docs/week01/README.md) |
| 第 2 周 | Token、Embedding、QKV 与 Self-Attention | [Attention](docs/week02/README.md) |
| 第 3 周 | 多头注意力、位置表示、FFN、残差与 LayerNorm | [Transformer Block](docs/week03/README.md) |
| 第 4 周 | GPT、causal mask、下一 token 预测、训练与采样 | [GPT、训练与生成](docs/week04/README.md) |
| 第 5～8 周 | 工具循环、上下文、检索、记忆、规划与应用原型 | [Agent 课程学习计划](docs/Agent课程学习计划.md) |
| 第 9～12 周 | 固定评测、对照实验、安全与项目报告 | [Agent 课程学习计划](docs/Agent课程学习计划.md) |
| 后续选学 | SFT、Agent RL 与研究实验 | [训练选学安排](docs/Agent课程学习计划.md#后续选学-agent-模型训练) |

Agent 阶段采用 **11-768 视频学习原理，CS329Z 作业练习系统开发** 的路线。先实现单 Agent 和固定评测，再按实验需要增加多 Agent 或模型训练。详细计划包含每周任务、完成标准、视频入口与复盘清单；第 5 周之后的材料和目录按实际进度创建。

课程资源：[CMU 11-768 课表与视频](https://www.cmu-agents.com/#/schedule)、[Stanford CS329Z 课表与幻灯片](https://cs329z.stanford.edu/index.html#schedule)、[CS329Z 作业 1](https://github.com/cs329z/assignment1-harness)。两门课程为 2026 秋季课程，材料随授课进度更新；录像访问情况详见学习计划。

## 目录结构

```text
ai_basic/
├── README.md                 # 项目入口、环境配置与目录约定
├── requirement.txt           # Python 与 Notebook 依赖
├── docs/                     # 学习文档：概念讲解、公式、课程说明
│   ├── 学习计划.md           # 总体路线与验收标准
│   ├── Agent课程学习计划.md  # 第 5～12 周安排、视频与实践任务
│   └── week01/ ～ week04/    # 已准备的基础阶段课程文档
├── exercises/                # 可独立运行的 Python 练习与示例
│   └── week01/
├── notebooks/                # 交互实验、计算过程与可视化
│   └── week01/
├── progress/                 # 学习进度、错题与每周复盘
├── outputs/                  # 运行生成的图表、模型等，不提交 Git
└── .venv/                    # 本地 Python 虚拟环境，不提交 Git
```

- 每周统一使用 `week01`、`week02` 等目录名，按学习进度创建。
- 同一课使用相同编号，例如 `docs/week01/01_vectors_and_matrices.md`、`exercises/week01/01_vectors_and_matrices.py`。
- Notebook 用于边运行边观察；独立脚本放在 `exercises/`，避免维护两份相同代码。
- 学习笔记和复盘放在 `progress/week01.md`；生成的文件放在 `outputs/week01/`。
- 后续有 Agent 完整项目时再创建 `projects/`，有外部数据时再创建 `data/`。
- 从项目根目录运行代码，例如 `.venv/bin/python exercises/week01/01_vectors_and_matrices.py`。

## 创建 Python 虚拟环境

在项目根目录执行以下命令，创建名为 `.venv` 的虚拟环境：

```bash
python3 -m venv .venv
```

### 激活虚拟环境

macOS / Linux（zsh、bash）：

```bash
source .venv/bin/activate
```

Windows（PowerShell）：

```powershell
.venv\Scripts\Activate.ps1
```

Windows（命令提示符）：

```cmd
.venv\Scripts\activate.bat
```

激活后，安装项目依赖（JupyterLab、Python 内核、NumPy 和 Matplotlib）：

```bash
python -m pip install --upgrade pip -i https://mirrors.ustc.edu.cn/pypi/simple
python -m pip install -r requirement.txt -i https://mirrors.ustc.edu.cn/pypi/simple
```


## 安装torch

通过代理下载
7897 仅为示例，请替换为代理软件实际的 HTTP 或混合端口
```bash
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --proxy http://127.0.0.1:7897 --timeout 120 --retries 10 --resume-retries 20

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --proxy http://127.0.0.1:7897 --timeout 120 --retries 10 --resume-retries 20

# cuda 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130  --proxy http://127.0.0.1:7897 --timeout 120 --retries 10 --resume-retries 20
```

验证安装成功：
```bash
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.version.cuda); print('cuDNN:',torch.backends.cudnn.version()); print('GPU:',torch.cuda.is_available())"
```


退出虚拟环境：

```bash
deactivate
```

## 启动与运行 Notebook

macOS 用户完成虚拟环境创建和依赖安装后，可直接双击项目根目录的 [start_notebook.command](start_notebook.command)，或在终端中运行：

```bash
./start_notebook.command
```

脚本自动定位项目根目录，使用 `.venv/bin/python` 启动 JupyterLab，无需手动激活虚拟环境。默认仅监听本机地址 `127.0.0.1`，启动后自动打开浏览器；请保持终端窗口运行。若提示没有执行权限，先执行 `chmod +x start_notebook.command`。可以传入额外参数，例如 `./start_notebook.command --no-browser --port=8889`。

Windows 用户完成虚拟环境创建和依赖安装后，可直接双击项目根目录的 [start_notebook.bat](start_notebook.bat)，或在 PowerShell 中运行：

```powershell
.\start_notebook.bat
```

脚本自动定位项目根目录，使用 `.venv\Scripts\python.exe` 启动 JupyterLab，无需手动激活虚拟环境。默认仅监听本机地址 `127.0.0.1`，启动后自动打开浏览器；请保持脚本窗口运行。若环境或依赖缺失，脚本会提示对应的安装命令。

也可以在项目根目录打开终端，激活 `.venv` 并完成上述依赖安装后，手动启动 JupyterLab：

```bash
python -m jupyterlab
```

保持终端运行，在自动打开的浏览器页面中进入 `notebooks/week01/`。如果浏览器没有自动打开，使用终端输出的本地访问链接（包含 token）。

### 创建和运行

1. 在目标目录点击 Launcher 中的 **Python 3 (ipykernel)**，创建 Notebook；已有 `.ipynb` 文件可直接双击打开。
2. 按课程编号保存文件，例如 `01_vectors_and_matrices.ipynb`。
3. 在代码单元格中输入代码，按 **Shift + Enter** 运行并进入下一格；按 **Ctrl + Enter** 运行当前格。
4. 按从上到下的顺序运行。需要验证整个文件时，使用 **Kernel → Restart Kernel and Run All Cells**，避免依赖之前运行遗留的变量。
5. 按 **Ctrl + S**（macOS 为 **Command + S**）保存。学习记录另写到 `progress/week01.md`。

可以用下面的单元格确认 Python 环境和数值计算正常：

```python
import sys
import numpy as np

print(sys.executable)  # 应指向项目 .venv 中的 Python
print(np.array([2, 3, 4]) @ np.array([1, 0, 2]))  # 10
```

如果内核使用了其他 Python 环境，在激活 `.venv` 的终端中注册项目内核：

```bash
python -m ipykernel install --user --name ai-basic --display-name "Python (ai_basic)"
```

然后在 Notebook 的内核选择器中切换到 **Python (ai_basic)**。

### 停止服务

先保存 Notebook，再回到启动 JupyterLab 的终端，按 **Ctrl + C**，按提示确认关闭服务。关闭浏览器页面不会停止服务。最后执行 `deactivate` 退出虚拟环境。
