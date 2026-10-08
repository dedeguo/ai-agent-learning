# AI Basic

学习路线：最小数学与神经网络 → Attention → Transformer → GPT → Agent 与评测。

## 目录结构

```text
ai_basic/
├── README.md                 # 项目入口、环境配置与目录约定
├── requirement.txt           # Python 与 Notebook 依赖
├── docs/                     # 学习文档：概念讲解、公式、课程说明
│   ├── 学习计划.md           # 总体路线与验收标准
│   └── week01/               # 第一周课程文档
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

从[总体学习计划](docs/学习计划.md)查看路线，从[第一周安排](docs/week01/README.md)开始学习。第二周进入 [Token、Embedding 与 Self-Attention](docs/week02/README.md)。

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
```
退出虚拟环境：

```bash
deactivate
```

## 启动与运行 Notebook

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

第三周进入 [Transformer Block](docs/week03/README.md)：多头注意力、位置信息、FFN、残差与 LayerNorm。
