# GuardedPy

GuardedPy is a local, governed coding-agent harness for small Python and pytest repositories. Its own loop builds context, requests one structured LLM action, applies deterministic policy, runs restricted tools, returns objective feedback, and stops on a bounded result. It also includes a separate, fixed-scenario demo that requires neither a project nor a credential.

## Installation

GuardedPy requires Python 3.11+ and a working operating-system keyring for local provider use. From a source checkout, create an environment and install the development dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

To test the distribution form after `make build`, install the wheel with `pipx install dist/guardedpy-*.whl`. This repository does not claim that GuardedPy is published to PyPI.

## Local operation

Run the complete local harness with:

```bash
guardedpy serve
```

Local mode binds to `127.0.0.1` only. The setup page asks for a target project, source/test directories, a pytest command, model name, timeout, and a hidden DeepSeek key. It can use a real DeepSeek-compatible provider only after setup; it does not use a `.env` file or any environment-variable fallback for credentials.

## Demo operation

Run the isolated mechanism demo with:

```bash
guardedpy demo
```

Demo mode is a distinct FastAPI factory with three fixed, offline mock-LLM scenarios: dangerous-action denial, failure-feedback correction, and TDD source-patch denial. It exposes neither target-project setup, keyring access, a real provider, arbitrary tasks, approvals, nor persistent state.

## Keyring lifecycle

The local setup form writes its submitted key directly to the operating-system keyring; the credentials page reports only configured/not-configured status and never displays the key. `CredentialService` implements deterministic set and clear operations, but this WebUI revision has no user-facing update or clear control. Replace an existing key by completing setup again; use your platform's keyring tooling to remove it. A missing or unavailable keyring is reported as an error—there is no plaintext fallback.

## Safety boundaries

The harness treats LLM output and repository text as untrusted. Deterministic policy code enforces repository boundaries, read-before-patch, TDD state transitions, restricted pytest execution, dangerous-action denial, and exact one-time approvals. It is not an OS sandbox: a user-selected repository and its tests remain trusted inputs, and malicious pytest code is out of scope.

## Tests and build

All verification commands are local and use no real LLM or key:

```bash
make test
make demo
make build
```

`make test` runs the offline pytest suite. `make demo` executes the three literal demo scenarios and exits nonzero if their statuses differ from the expected deterministic result. `make build` runs `python -m build --no-isolation` to create a wheel and sdist in `dist/`.

## Render demo wake-up

`render.yaml` is a demo-only deployment blueprint: its start command serves `guardedpy.demo:create_demo_app`, never the keyring-backed local application. The blueprint is not deployed and this README intentionally contains no Render URL. If it is deployed on a free Render instance, the first request after an idle period may wait while the instance wakes; verify the generated service URL in Render before publishing it here.

## CI evidence

The [GitHub Actions workflow](.github/workflows/ci.yml) and [.gitlab-ci.yml](.gitlab-ci.yml) install the same development dependencies and run `make test`, `make demo`, and `make build`; the GitLab job is named exactly `unit-test`. GitHub Actions has recorded successful validation runs during the reconstructed PR sequence; the repository Actions and PR pages are the authoritative current evidence. The GitLab workflow has not yet been executed, so no GitLab pass is claimed.

## Open Design attribution

The local templates and CSS follow the selected [Open Design Agentic direction](https://open-design.ai/plugins/design-system-agentic/): clear task outcomes, restrained controls, semantic status badges, and readable local-console surfaces. GuardedPy uses handwritten Jinja, CSS, and JavaScript; it does not bundle Open Design assets. See the [Open Design project](https://open-design.ai/) for its local-first, BYOK design workflow and licensing information.

## Known limitations

- Local mode supports one active task and Python 3.11+ projects that use pytest; it is not a multi-user, remote, or multi-project runner.
- The public demo is evidence of fixed mechanisms, not a hosted version of the local coding harness.
- Local mode needs a functioning OS keyring. The current WebUI lacks a visible credential update/clear control, as noted above.
- The Render blueprint is deployment preparation only: it has not been deployed, and no public URL is available yet.
- GitHub Actions has been exercised; GitLab CI and the Render deployment have not.
