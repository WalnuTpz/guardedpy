# GuardedPy

GuardedPy is a local, governed coding-agent harness for small Python and pytest repositories. Its self-owned continuous loop keeps one session and one active turn, streams model text and governed tool status, feeds concise pytest results back to the same turn, and restores visible user/Agent conversation plus safe tool facts across restarts. It is a CLI-only Python package: start it from the project you want to inspect or change.

## Installation

GuardedPy supports Python 3.11+ on Linux and WSL. Install from a source checkout for development:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

To verify the distributable form locally, build it first and install the generated wheel into a separate environment:

```bash
make build
python3 -m venv /tmp/guardedpy-release-env
/tmp/guardedpy-release-env/bin/python -m pip install dist/guardedpy-*.whl
```

The release artifact is **not uploaded** to a hosting platform yet. Until a real release is published, use this source checkout or a wheel built by the commands above; this README deliberately provides no invented download link.

## Run in a project

Move into a supported Python + pytest repository and launch the one command:

```bash
cd /path/to/python-pytest-project
guardedpy
```

GuardedPy discovers the current project, its common source/test layout, and the full pytest command automatically. There is no manual root, source-directory, test-directory, or pytest-command setup flow. In an interactive terminal, enter a natural-language coding task or choose a command from the live command palette. A short initial task may also be passed directly:

```bash
guardedpy "find the failing tests and explain a safe repair plan"
```

When stdin or stdout is redirected, GuardedPy uses a safe plain-text session. It cannot accept secret input there and stops rather than auto-approving a dangerous action.

The interactive command set is:

```text
/conversations：选择并恢复历史对话
/new：新建会话
/delete：删除当前会话并回到上一条
/exit：退出 GuardedPy
/plan <任务>：只读制定计划
/review <路径>：只读审查
/goal <目标>：只约束下一回合
/queue <任务>：将下一项工作排队
/stop：中断当前回合
/tests：运行配置的 pytest
/diff：查看当前 Git diff
/permissions：查看自动允许与须审批的操作
/doctor：查看本地项目状态
/credentials：管理 API Key
/model：选择后续回合模型
/effort：选择后续回合思考强度
/help：打开完整帮助
```

直接输入自然语言即可开始同一连续会话；不再对首条输入做 feature/bugfix/闲聊分类。启动时会自动载入最近保存的会话，首次启动才新建空会话。`/plan <request>` 与 `/review [path]` 是只读回合；`/goal <目标>` 仅约束下一回合；`/queue` 显式排队下一回合，`/stop` 请求中断当前回合。工具失败、pytest failure 和无效 patch 都会回灌给同一回合，而不是自动取消。Agent 可在不经过 shell 的前提下运行项目内单个 Python 程序；网络、安装、任意命令和 Git 写入仍不可由它执行。transcript 是自动换行、仅纵向滚动的只读文本；鼠标可以选择它，`Ctrl+Shift+C` 可复制当前安全记录。`/conversations` 会重放所选历史的可见用户/Agent 对话，并以它们作为后续模型上下文；`/delete` 删除当前会话后回到上一条，最后一条不能删除。

## Credentials, model, and effort

In an interactive session, `/credentials` opens a masked input for the DeepSeek API key. The value is written only through the operating-system keyring, is not echoed in the transcript, and can be updated or cleared through the same controlled UI. Before any coding, plan, or review task starts, GuardedPy requires this interactive credential step; redirected sessions stop with an explicit message and never create the task. Do not put a key in project files, command arguments, environment variables, logs, or examples. An unavailable keyring fails safely; there is no plaintext fallback.

The default is `deepseek-v4-flash` with `high` effort. In the interactive session, `/model` or `/effort` opens a keyboard- and mouse-selectable picker; the selected value applies only to later tasks. In redirected plain text, provide the supported value explicitly:

```text
/model deepseek-v4-flash
/model deepseek-v4-pro
/effort high
/effort max
```

An active task retains its creation snapshot. `max` effort can increase latency and provider cost.

## Mechanism demo

The fixed demo is offline and does not inspect the caller project, touch the keyring, or call a provider:

```bash
guardedpy demo
make demo
```

Interactive `guardedpy demo` reuses the normal project bar, transcript, composer frame, `[+]` selector, and approval dialog. Its header shows `项目：机制演示临时项目` plus the right-aligned hint `按↑↓来切换场景`; the composer provides visual-only selectors for `Mock LLM1/2`, `high/max`, and `[发送]`. Those selectors never change the fixed scenario or mock event sequence. ↑/↓ switches the fixed request. `[+]` opens the normal command palette, whose choices intentionally have no effect in this locked demonstration. Press Enter or `[发送]` to run the selected fixed request. The request then drives the actual `ConversationAgent` event loop; the transcript displays its governed tool, approval, feedback, and final-reply events rather than a prewritten chat record. `make demo` runs all three deterministic mock-LLM scenarios without interaction and asserts their mechanism facts:

1. 已读取项目文件后的删除请求暂停并显示审批；交互演示中可允许或拒绝，`make demo` 固定拒绝并验证目标文件保留；
2. assertion feedback 回灌后 mock 返回修复补丁，随后 pytest 通过；
3. 伪造或过期 approval ID 被确定性拒绝。

This is mechanism evidence, not a substitute for using a real provider on a target project.

## Manual provider acceptance (not yet claimed complete)

After configuring a DeepSeek key in the interactive `/credentials` dialog, use a disposable Python + pytest project and verify the real provider in this order:

1. Send a normal greeting and a project question; both should receive a natural response without starting a form or a fixed task workflow.
2. Send a repair request such as “找出并修复所有测试错误”; the transcript should stream the user message, assistant text, safe read/test/change status, and a final response.
3. Ask “刚才改了什么，测试结果怎样？” in the same session; the reply should use the preceding tool facts.
4. Ask to delete a previously read project file; verify that the approval dialog names that file, then reject it and confirm the file remains.
5. Start a longer request, use `/stop`, and verify that the turn becomes interrupted only after its terminal event. Restart `guardedpy`, use `/conversations`, and confirm that visible user/Agent dialogue returns and can ground a follow-up, while source, raw diffs, tool arguments and hidden reasoning do not return.

Do not paste API keys into chat, project files, command arguments, environment variables, or screenshots. Record this manual result only after performing it; it is intentionally not claimed by this repository yet.

## Test, build, and local release artifact

All repository checks are local and use mock/stub LLMs rather than a real key or network call:

```bash
make test PYTHON=python
make demo PYTHON=python
make build PYTHON=python
```

`make test` runs the offline test suite. `make demo` asserts the stable safe summaries for the three mechanism scenarios. `make build` creates a wheel and source distribution in `dist/`. The CI definitions use these same three commands; the GitLab job is named `unit-test`. Remote CI, a release upload, GitLab mirroring, collaborator access, and the student's own reflection remain external handoff steps and are not claimed complete here.

## Directory structure

```text
.
├── src/guardedpy/          # self-owned loop, policy, tools, feedback, and terminal client
├── tests/                  # offline unit, integration, terminal, and artifact contracts
├── scripts/                # non-interactive mechanism demonstration runner
├── docs/                   # course requirements, design records, and implementation plans
├── Makefile                # test, demo, and build entry points
├── pyproject.toml          # package metadata and sole `guardedpy` console entry point
├── .github/workflows/ci.yml
└── .gitlab-ci.yml
```

## Safety boundaries and limitations

GuardedPy treats LLM output and repository text as untrusted. Deterministic code enforces project-root boundaries, read-before-patch, atomic patch application, restricted pytest invocation, exact delete approvals, safe summary persistence, and bounded turn/tool execution. The harness is not an operating-system sandbox: a selected repository and its pytest code are trusted inputs, so malicious tests are out of scope.

The first release supports one active task at a time and Python 3.11+ pytest projects on Linux or WSL. It does not support Windows-native execution leases, multi-project operation, remote workspaces, arbitrary shell access, deployment, browser delivery, or an HTTP interface. Textual needs an interactive TTY; redirected I/O has the deliberately narrower safe text mode.

The project owner reports a teacher clarification permitting a CLI-only release artifact. The general course materials still describe a browser-interface default, so that clarification must be retained as real submission evidence; this repository does not represent the CLI-only form as satisfying that default without it.

## Third-party software and licenses

GuardedPy declares these direct dependencies and imports their installed distributions rather than copying their source: Pydantic (MIT), keyring (MIT), OpenAI Python SDK (Apache-2.0), pytest (MIT), PyYAML (MIT), Textual (MIT), build (MIT), and setuptools (MIT). Distributors should check the resolved package metadata for the versions they actually ship.
