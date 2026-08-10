# GuardedPy

GuardedPy 是一个面向小型 Python + pytest 项目的本地 CLI Coding Agent Harness。它自行实现连续 Agent 主循环，让用户在同一终端会话中自然对话、检查项目、修改代码、运行测试并继续追问；文件边界、测试反馈、危险动作审批、凭据和会话恢复由确定性代码控制。

当前产品只提供 CLI，不包含 WebUI、HTTP server，也不向模型提供任意 Shell、通用网络、依赖安装或自动发布工具。

> 本仓库是 GuardedPy 的公开展示与 Release 分发镜像，提供当前可运行的源码、测试、构建配置和发布资产。
>
> 课程过程文档、开发日志与学生反思仅保存在供课程评审访问的私有课程仓库；本公开仓库不包含这些材料。

## 核心功能

- 从当前目录自动发现 Python 源码、测试目录和 pytest 命令；
- 连续 Session/Turn 对话，不对首条输入进行固定任务分类；
- 流式显示模型回复和安全工具状态；
- 受限文件列表、读取、原子补丁和源码/测试新文件创建；
- pytest 反馈分类、回灌和修改后的完整套件验证；
- 项目内 Python 程序运行与文件删除的精确人工审批；
- 只读计划、审查、Git status/diff 和本地诊断；
- keyring-only DeepSeek API Key 管理；
- 会话保存、恢复、选择与删除；
- 离线 mock LLM 单元测试和可观看的机制演示；
- wheel/sdist 构建及 GitHub/GitLab CI。

## 支持环境

- Linux 或 WSL；
- Python 3.11+；
- 目标项目使用 pytest；
- 交互 TUI 需要真实终端；
- 真实 DeepSeek 调用需要可用的系统 keyring/Secret Service 和到 provider 的网络连接。

## 安装

### 从源码安装

```bash
git clone git@github.com:WalnuTpz/guardedpy.git
cd guardedpy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### 从本地构建产物安装

```bash
make build
python3 -m venv /tmp/guardedpy-release-env
/tmp/guardedpy-release-env/bin/python -m pip install dist/guardedpy-*.whl
```

当前稳定版本为 [v0.1.0](https://github.com/WalnuTpz/guardedpy/releases/tag/v0.1.0)，可直接安装 [wheel](https://github.com/WalnuTpz/guardedpy/releases/download/v0.1.0/guardedpy-0.1.0-py3-none-any.whl)。

## 运行

进入需要处理的 Python + pytest 项目，直接启动：

```bash
cd /path/to/python-project
guardedpy
```

GuardedPy 会把当前目录固定为项目根，自动发现源码、测试和 pytest 配置。无需 `/init`，也不要求手工输入目录或测试命令。

可以传入一条初始请求：

```bash
guardedpy "检查当前项目并说明测试失败原因，不要修改"
```

stdin 或 stdout 被重定向时，GuardedPy 使用安全纯文本模式。该模式不能录入秘密或批准危险操作；遇到审批会停止。

## 凭据配置

在交互界面输入：

```text
/credentials
```

凭据界面支持：

- 查看是否已配置；
- 掩码录入或更新 DeepSeek API Key；
- 清除 Key。

Key 只写入操作系统 keyring，不会进入项目文件、Git、会话数据库、日志或终端输出。GuardedPy 不提供 `.env`、命令行参数或明文配置回退。若 keyring 不可用，程序会说明需要先修复系统密钥环。

不要把 API Key 粘贴到对话、命令、项目文件、截图或测试中。

## 交互方式

- Enter：提交消息；
- Shift+Enter：输入换行；
- `/` 或 `[+]`：打开命令面板；
- 方向键/鼠标滚轮：移动选择；
- Enter 或鼠标点击：把候选项填入输入框；
- `Ctrl+Shift+C`：复制当前安全 transcript；
- Ctrl+C、Esc 或 `/stop`：请求中断当前回合。

键盘可以完成全部操作，鼠标用于辅助选择、滚动和点击。transcript 自动换行，只保留纵向滚动条。

## 命令

| 命令 | 作用 |
|---|---|
| `/conversations` | 选择并恢复已保存会话 |
| `/new` | 新建会话 |
| `/delete` | 删除当前会话并切回上一条；最后一条不能删除 |
| `/plan <任务>` | 用只读模式制定计划 |
| `/review <路径>` | 用只读模式审查项目或路径 |
| `/goal <目标>` | 设置只影响下一回合的目标；`/goal clear` 取消 |
| `/queue <任务>` | 把下一回合排到当前回合之后 |
| `/stop` | 中断当前回合并清除排队回合 |
| `/tests` | 运行发现到的 pytest 套件 |
| `/diff` | 查看当前 Git diff |
| `/permissions` | 查看自动允许、须审批和拒绝的操作 |
| `/doctor` | 查看当前项目、配置和凭据状态 |
| `/credentials` | 安全管理 API Key |
| `/model` | 为后续回合选择模型 |
| `/effort` | 为后续回合选择思考强度 |
| `/help` | 打开分组帮助面板 |
| `/exit` | 退出 GuardedPy |

支持的模型为 `deepseek-v4-flash`、`deepseek-v4-pro`；思考强度为 `high`、`max`。修改只影响后续回合，不改变正在运行的 Turn。

## 工具和审批

模型只能调用 GuardedPy 声明的受限工具：

- 列出和读取项目文件；
- 对源码/测试应用原子补丁；
- 运行 pytest；
- 读取 Git status/diff；
- 运行一个项目内 Python 文件；
- 删除一个普通文件或空目录。

读取、合法补丁、pytest 和 Git 只读检查经过确定性校验后自动执行。以下操作每次都需要精确审批：

- 运行项目内 Python 程序：审批框显示路径和参数；
- 删除文件或空目录：审批框显示目标路径。

拒绝、过期或伪造的审批不会执行动作。GuardedPy 不向模型提供任意 Shell、网络、包安装、Git 写入、部署或发布工具。

## 会话恢复

无初始任务启动时，GuardedPy 自动载入当前项目最近的会话。`/conversations` 可选择其他会话，`/delete` 可删除当前会话。

会话状态位于项目目录之外、按项目根隔离的应用状态目录。可恢复内容包括可见用户/Agent 对话和受限工具事实。以下内容不会保存或恢复：

- API Key；
- 隐藏 reasoning；
- 源码正文；
- 完整 diff；
- 原始工具参数；
- 完整 pytest 输出；
- 中断进程的执行状态。

## 机制演示

### 交互式演示

```bash
guardedpy demo
```

演示使用正常 TUI 外壳，但项目、请求和 mock 输出均为隔离的固定数据。使用 ↑/↓ 切换场景，按 Enter 或 `[发送]` 开始。删除场景会显示真实审批弹层，可以允许或拒绝。

### 无交互演示

```bash
make demo
```

该命令使用同一 `ConversationAgent` 路径，确定性验证三个场景：

1. 删除操作暂停等待审批；headless 模式固定拒绝并验证文件仍存在；
2. pytest assertion failure 回灌后，mock 改变下一步动作为补丁，最终测试通过；
3. 伪造或过期审批 ID 被拒绝。

演示不读取调用者项目、keyring，不访问网络，也不调用真实 provider。

## 测试与构建

```bash
make test
make demo
make build
python -m compileall -q src tests
git diff --check
```

- `make test`：运行全部离线单元与集成测试；
- `make demo`：运行固定 mock 机制证据；
- `make build`：在 `dist/` 生成 wheel 和 sdist。

离线测试使用 mock/stub LLM，不需要真实 Key 或网络。GitHub Actions 执行 test/demo/build 门禁；仓库同时附带 GitLab CI 配置，其中 job 名为 `unit-test`。

## 快速人工验收

可在一个带已知失败测试的临时项目中依次验证：

1. 普通问候和项目问题能自然回答；
2. “检查错误但不要修复”不会写文件；
3. 修复请求会读取、补丁、运行完整 pytest 并总结结果；
4. 新建文件只能位于发现的源码或测试目录；
5. 运行 Python 程序会显示准确审批，批准后显示有界输出；
6. 删除操作可先拒绝并确认文件保留；
7. `/stop` 能中断活动回合；
8. 重启后从 `/conversations` 恢复，并能基于此前事实继续追问。

## 目录结构

```text
.
├── src/guardedpy/
│   ├── conversation.py     # Session/Turn/Event 与连续主循环
│   ├── governor.py         # 工具治理与审批身份
│   ├── executor.py         # 工具分发和反馈回灌
│   ├── workspace.py        # 根目录受限文件、pytest、Python、Git 工具
│   ├── feedback.py         # pytest 确定性分类
│   ├── runtime.py          # 会话组成、持久化与恢复
│   ├── conversations.py    # 项目隔离的 SQLite 会话存储
│   ├── credentials.py      # keyring-only 凭据服务
│   ├── discovery.py        # Python/pytest 项目发现
│   ├── tui.py              # Textual 正常界面与演示界面
│   ├── terminal.py         # 安全纯文本入口
│   └── mechanism_demo.py   # 三个 scripted mock 场景
├── tests/                  # 离线机制、终端和安装产物测试
├── scripts/                # 无交互机制演示入口
├── Makefile
├── pyproject.toml
├── .github/workflows/ci.yml
└── .gitlab-ci.yml
```

## 安全边界

- LLM 输出、仓库内容和测试输出都按不可信数据处理；
- 所有路径在执行前重新解析并限制在项目根；
- 补丁只能修改发现到的源码/测试范围，失败时零写入；
- 子进程使用固定 argv、超时和 `shell=False`；
- 删除和程序运行使用绑定当前调用的精确审批；
- transcript 和摘要在持久化前脱敏并限制长度；
- 凭据只进入系统 keyring 和实际 provider 调用。

## 已知限制

- pytest 会执行项目代码；恶意仓库不在安全模型内；
- 只支持 Linux/WSL、Python 3.11+ 和 pytest 项目；
- 只支持一个本地项目和一个活动 Turn；
- 不支持 Windows 原生终端、远程工作区、非 pytest 测试框架或多 Agent；
- 不向模型提供任意 Shell、通用网络、安装、Git 写入、发布、部署或后台终端管理工具；
- TUI 需要交互终端；重定向模式功能更窄；
- keyring/Secret Service 不可用时不会回退到明文凭据；
- 恢复会话只重建有界对话和安全事实，不恢复模型进程全部状态。

## 分发状态

当前稳定发布为 [v0.1.0](https://github.com/WalnuTpz/guardedpy/releases/tag/v0.1.0)，提供 wheel 与 sdist 两种安装资产。

## 第三方依赖与许可证

GuardedPy 通过包管理器使用下列直接依赖，不复制其源码：Pydantic（MIT）、keyring（MIT）、OpenAI Python SDK（Apache-2.0）、pytest（MIT）、PyYAML（MIT）、Textual（MIT）、build（MIT）和 setuptools（MIT）。发布时应以实际构建环境中的包元数据为准复核版本与许可证。
