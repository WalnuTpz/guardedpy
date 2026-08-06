# GuardedPy

GuardedPy is a local, governed coding-agent harness for small Python and pytest repositories. Its self-owned loop creates context, asks one structured LLM action at a time, applies deterministic policy, runs restricted project tools, feeds back concise pytest evidence, and stops at a bounded result. It is a CLI-only Python package: start it from the project you want to inspect or change.

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
会话与对话：/history /new /clear /exit
任务与检查：/plan /review /tests /diff
设置与安全：/model /effort /permissions /credentials /memory /doctor
帮助：/help
```

`/plan <request>` and `/review [path]` are read-only workflows. `/tests`, `/diff`, and `/doctor` are deterministic local checks, not LLM requests. A coding task first runs the discovered full test suite: a passing baseline enters feature TDD, assertion failures enter the repair path, and collection/execution errors or a timeout block further automated changes.

## Credentials, model, and effort

In an interactive session, `/credentials` opens a masked input for the DeepSeek API key. The value is written only through the operating-system keyring, is not echoed in the transcript, and can be updated or cleared through the same controlled UI. Do not put a key in project files, command arguments, environment variables, logs, or examples. An unavailable keyring fails safely; there is no plaintext fallback.

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

Interactive `guardedpy demo` lets you select one fixed request and press Enter. `make demo` runs all three deterministic mock-LLM scenarios without interaction:

1. a privileged command request is denied before command dispatch;
2. assertion feedback is required before the mock returns a repair patch, then the full suite passes;
3. a production patch before an observed red test is denied.

This is mechanism evidence, not a substitute for using a real provider on a target project.

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

GuardedPy treats LLM output and repository text as untrusted. Deterministic code enforces project-root boundaries, read-before-patch, baseline/TDD transitions, restricted pytest invocation, explicit approvals, and bounded task execution. The harness is not an operating-system sandbox: a selected repository and its pytest code are trusted inputs, so malicious tests are out of scope.

The first release supports one active task at a time and Python 3.11+ pytest projects on Linux or WSL. It does not support Windows-native execution leases, multi-project operation, remote workspaces, arbitrary shell access, deployment, browser delivery, or an HTTP interface. Textual needs an interactive TTY; redirected I/O has the deliberately narrower safe text mode.

The project owner reports a teacher clarification permitting a CLI-only release artifact. The general course materials still describe a browser-interface default, so that clarification must be retained as real submission evidence; this repository does not represent the CLI-only form as satisfying that default without it.

## Third-party software and licenses

GuardedPy declares these direct dependencies and imports their installed distributions rather than copying their source: Pydantic (MIT), keyring (MIT), OpenAI Python SDK (Apache-2.0), pytest (MIT), PyYAML (MIT), Textual (MIT), build (MIT), and setuptools (MIT). Distributors should check the resolved package metadata for the versions they actually ship.
