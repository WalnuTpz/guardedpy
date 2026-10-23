# 安装并完成第一次修复

这篇教程从一个有缺陷的分组函数开始：`chunks([1, 2, 3, 4, 5], 2)` 应返回 `[[1, 2], [3, 4], [5]]`，原实现却丢掉了末尾的 `[5]`。你将创建项目和测试，让 GuardedPy 完成真实修复，再检查代码与测试结果。

准备 Linux 或 WSL、Python 3.11+、Git 和交互终端，以及 DeepSeek API Key 和到 `https://api.deepseek.com` 的网络连接。下面会依次准备 Python 环境、示例项目和保存凭据的系统密钥环。

## 1. 安装到示例环境

将 GuardedPy 源码和示例放在 `~/projects`：

```bash
mkdir -p ~/projects
git clone https://github.com/WalnuTpz/guardedpy.git ~/projects/guardedpy
mkdir ~/projects/guardedpy-example
cd ~/projects/guardedpy-example
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ~/projects/guardedpy
guardedpy --version
```

版本命令输出 `guardedpy 0.1.0`。pytest 随 GuardedPy 安装。接下来的命令都在这个环境和示例目录中运行。

## 2. 用测试复现问题

创建 `batching.py` 和四个测试：

```bash
mkdir tests
cat > batching.py <<'PYTHON'
def chunks(items, size):
    return [items[start:start + size]
            for start in range(0, len(items) - size + 1, size)]
PYTHON
cat > tests/test_batching.py <<'PYTHON'
from batching import chunks


def test_exact_groups():
    assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_partial_last_group():
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_smaller_than_group():
    assert chunks([1], 3) == [[1]]


def test_empty_input():
    assert chunks([], 2) == []
PYTHON
python -m pytest -q
```

pytest 输出 `2 failed, 2 passed`。`test_partial_last_group` 发现末尾元素被丢弃；`test_smaller_than_group` 发现不足一组的输入被整个跳过。完整分组和空输入两个用例已经通过。

此时项目包含：

```text
guardedpy-example/
├── batching.py
└── tests/
    └── test_batching.py
```

根目录的 Python 模块和 `tests/` 目录让 GuardedPy 能够识别这个项目。四个用例共同定义修复目标：保留尾组，同时保持完整分组和空输入的行为。

## 3. 准备密钥环并连接模型

GuardedPy 通过系统 keyring 保存和读取 DeepSeek API Key。先在示例虚拟环境中查看 Python 选中的后端：

```bash
python -c "import keyring; print(keyring.get_keyring())"
```

`keyring.backends.SecretService.Keyring` 表示使用 Secret Service；`keyring.backends.fail.Keyring` 表示当前环境尚无可用后端。WSL 中的 Linux Python 使用 Linux 密钥环服务。

### 桌面用户会话

Secret Service 通过用户的 D-Bus 会话访问密钥环。在桌面环境中，用系统密钥环管理器解锁密码存储，再从同一桌面用户会话打开终端，进入示例目录并激活 `.venv`。已有可用后端和已解锁密钥环时，可以直接保存 API Key。

### 只有命令行的 Ubuntu 或 WSL

安装服务并进入一个 D-Bus 会话：

```bash
sudo apt install dbus gnome-keyring
dbus-run-session -- bash
```

在新打开的 Bash 中，用隐藏输入读取登录密钥环的密码，再解锁或创建密钥环：

```bash
read -rsp '密钥环密码: ' guardedpy_keyring_password
printf '\n'
printf '%s' "$guardedpy_keyring_password" | gnome-keyring-daemon --unlock
unset guardedpy_keyring_password
```

这里填写的是登录密钥环的密码；首次创建时，该输入成为它的密码。服务随后在后台运行。保持在这个 Bash 中，返回示例环境：

```bash
cd ~/projects/guardedpy-example
source .venv/bin/activate
python -c "import keyring; print(keyring.get_keyring())"
```

以后使用时，重新进入 D-Bus 会话并解锁已有密钥环，再从该会话启动 GuardedPy。

### 保存 API Key

启动终端界面：

```bash
guardedpy
```

输入 `/credentials`，在掩码输入框中填入 DeepSeek API Key，选择“更新”。看到“凭据已更新”后，即可提交任务。同一用户环境中的项目共用这个凭据条目。

## 4. 让 Agent 修复实现

在主输入框输入以下任务，按 Enter 提交：

```text
修复 chunks 丢失末尾不足一组元素的问题。保留函数签名和四个测试的预期，先检查失败原因，修改实现后运行完整 pytest，并说明修复结果。
```

一次提交启动一个回合，模型可以在其中多次读取文件、修改和测试。观察终端中的文件读取、补丁和 pytest 结果，直到出现最终回复。这个项目的完成结果应包含对 `batching.py` 的修改，以及完整测试的 `4 passed`。

模型请求删除文件或运行 Python 程序时，界面会显示具体操作和参数；选择“本次允许”执行该调用，或选择“拒绝”将决定交回模型。

## 5. 检查代码与结果

输入 `/exit` 返回终端，查看文件并重新运行测试：

```bash
cat batching.py
python -m pytest -q
```

一种符合要求的实现是：

```python
def chunks(items, size):
    return [items[start:start + size]
            for start in range(0, len(items), size)]
```

原来 `range` 的终点是 `len(items) - size + 1`，只会枚举能形成完整分组的起点。改成 `len(items)` 后，最后一个起点也进入循环，Python 切片会取出余下的元素。对 `[1, 2, 3, 4, 5]`，起点依次是 `0`、`2`、`4`，对应三个分组。

确认四个测试的输入和预期保持原样，pytest 输出 `4 passed`。这样既验证了尾组和短输入的修复，也检查了原本正确的两种行为。

界面若显示“本轮未完成：修改后尚未通过完整 pytest 验证”，文件会保留当前内容。查看最后的测试结果，再提交“检查当前修改，运行完整 pytest，并修复仍然失败的用例”继续处理。

完成这次修复后，可以在同一会话追问实现细节。接入自己的项目、计划、审查和会话操作见[使用手册](usage.md)。
