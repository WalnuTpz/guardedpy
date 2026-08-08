PYTHON ?= python

.PHONY: test demo build

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests -q

demo:
	PYTHONPATH=src $(PYTHON) scripts/run_mechanism_demo.py

build:
	$(PYTHON) -m build --no-isolation
