PYTHON ?= python

.PHONY: test demo demo-assets build cli-check

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests -q

demo:
	PYTHONPATH=src $(PYTHON) -c 'from guardedpy.demo import run_scenario; from guardedpy.domain import TaskStatus; expected = {"dangerous_action_denied": TaskStatus.BLOCKED, "failure_feedback_corrects": TaskStatus.COMPLETED, "tdd_source_patch_denied": TaskStatus.BLOCKED}; actual = {name: run_scenario(name).status for name in expected}; assert actual == expected, actual'

demo-assets:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_demo.py tests/test_frontend_ui.py -q

build:
	$(PYTHON) -m build --no-isolation

cli-check:
	PYTHONPATH=src $(PYTHON) -c 'from guardedpy.cli import main; raise SystemExit(main(["--help"]))'
	PYTHONPATH=src $(PYTHON) -c 'from guardedpy.cli import main; raise SystemExit(main(["--help"]))'
	PYTHONPATH=src $(PYTHON) -c 'from guardedpy.cli import server_main; raise SystemExit(server_main(["--help"]))'
