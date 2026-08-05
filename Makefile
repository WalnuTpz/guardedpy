PYTHON ?= python

.PHONY: test demo build

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests -q

demo:
	PYTHONPATH=src $(PYTHON) -c 'from guardedpy.demo import run_scenario; from guardedpy.domain import TaskStatus; expected = {"dangerous_action_denied": TaskStatus.BLOCKED, "failure_feedback_corrects": TaskStatus.COMPLETED, "tdd_source_patch_denied": TaskStatus.BLOCKED}; actual = {name: run_scenario(name).status for name in expected}; assert actual == expected, actual'

build:
	$(PYTHON) -m build --no-isolation
