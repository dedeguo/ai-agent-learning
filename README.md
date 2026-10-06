# AI Basic

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

激活后，可以用 pip 安装项目依赖：

```bash
python -m pip install --upgrade pip
# 示例：python -m pip install <package-name>
```

退出虚拟环境：

```bash
deactivate
```
