# Codex Desktop Windows 沙箱执行故障排查与解决

> 记录日期：2026-10-10

## 一、现象

Codex Desktop（Windows）无法执行任何本地命令，所有 shell、Git 与 Node 内核在真正执行前即失败，报错为：

```
Failed to create unified exec process: helper_unknown_error: setup refresh had errors
```

具体表现：

- PowerShell、cmd、git 等命令全部无法启动，失败发生在命令执行之前，仓库与脚本均未被触碰。
- `node_repl` 内核同样报错并退出：

  ```
  windows sandbox failed: helper_unknown_error: setup refresh had errors
  kernel_status: exited(code=1)
  ```

- App 层功能（文档、PDF、MCP、依赖路径查询等）不受影响，说明故障集中在沙箱执行链路，而非 Codex 应用本体。
- 更换工作目录、shell、TTY 等均无效，属于全局性沙箱初始化失败。

## 二、根因

`setup refresh had errors` 是沙箱 helper 在刷新 / 初始化执行环境时失败时给出的**共用顶层报错**，背后至少存在两类不同原因，需结合沙箱日志中的错误码判断：

1. **目录 ACL 所有权异常（错误码 5）**
   仓库内的 `.git` 或 `.codex` 目录所有者被改成 `CodexSandboxOffline`，沙箱刷新 ACL 时写入被拒，日志出现 `SetNamedSecurityInfoW failed: 5`。

2. **运行时文件被占用（错误码 32）**
   捆绑的运行时（如 `node_repl.exe`）被其他进程占用，沙箱校验其权限时报 `file is being used by another process`，与第一类不是同一问题。

该问题在近期多个 Windows 版本（26.819.x / 26.820.x / 26.924.x 等）中均有报告，属于回归类故障。

## 三、定位方法

查看 `C:\Users\<用户名>\.codex\` 下的 sandbox 日志，找到具体错误码和被点名的目录：

- 错误码 `5`（`SetNamedSecurityInfoW failed: 5`）→ 方案 A。
- 错误码 `32`（文件被占用，涉及 `node_repl.exe`）→ 方案 B。

## 四、解决方案

### 方案 A：恢复目录所有权（对应错误码 5）

1. 彻底退出 Codex，并在任务管理器中结束所有残留的 `ChatGPT.exe`、`codex.exe` 进程；仅关闭窗口通常不够。
2. 以管理员身份打开 PowerShell，进入仓库目录，先确认所有者：

   ```powershell
   (Get-Acl ".git").Owner
   (Get-Acl ".codex").Owner
   ```

   若显示 `CodexSandboxOffline`，则恢复所有权（只处理日志中点名的目录即可）：

   ```powershell
   $me = "$env:USERDOMAIN\$env:USERNAME"
   icacls ".codex" /setowner $me /T /C
   icacls ".git" /setowner $me /T /C
   ```

3. 重新打开 Codex，命令执行即可恢复。

### 方案 B：降级为 unelevated 沙箱（对应错误码 32 / 通用 workaround，本次采用）

在 `%USERPROFILE%\.codex\config.toml` 中新增以下配置：

```toml
[windows]
sandbox = "unelevated"
```

彻底重启 Codex 后生效。

- 效果：shell 与 Node 内核恢复执行，本次问题解决。
- 代价：切换到更弱的隔离模型（非 elevated 沙箱），属于临时 workaround 而非根治，需持续关注官方修复版本后恢复默认沙箱。

## 五、本次处理记录

- 采用方案 B，配置路径为 `C:\Users\54204\.codex\config.toml`。
- 重启后验证：`echo`、目录读取等基础命令恢复正常，`exec_command` 可正常创建进程。

## 六、注意事项与后续

- 无论采用哪种方案，第一步都应彻底退出并清理后台进程。
- 优先将 Codex Desktop 更新到最新版本，确认是否为已修复的回归。
- 方案 B 会降低隔离强度，长期应换回官方默认沙箱，或在官方修复后移除 `sandbox = "unelevated"` 配置。

## 七、参考资料

- 社区讨论：[Codex Desktop command runner completely broken: helper_unknown_error: setup refresh had errors](https://community.openai.com/t/codex-desktop-command-runner-completely-broken-helper-unknown-error-setup-refresh-had-errors/1391898)
- GitHub Issue：`openai/codex#39841`、`openai/codex#40596`、`openai/codex#40215`