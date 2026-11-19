# Day-to-day targets. The interpreter is whatever `python` is on the path (activate your venv first) or PY=.
PY ?= python
TORCH_INDEX ?= cpu            # cpu or cu128

.PHONY: install test test-gpu lint clean-env figures tools help

help:
	@echo "install    pinned requirements (TORCH_INDEX=$(TORCH_INDEX)) and the package, editable"
	@echo "test       the CPU test suite"
	@echo "test-gpu   the two end-to-end checks against the mock server (CUDA, cached weights, WAYS_DATA)"
	@echo "lint       ruff over the package, the experiments and the tests"
	@echo "clean-env  the clean-environment execution test (fresh venv, pip check, tests, lock file)"
	@echo "figures    the hand-drawn figures and the two charts (Playwright Chromium)"
	@echo "tools      list the registered tools"

install:
	$(PY) -m pip install --extra-index-url https://download.pytorch.org/whl/$(TORCH_INDEX) -r requirements.txt
	$(PY) -m pip install --no-deps -e .

test:
	$(PY) -m pytest tests -q -m "not gpu and not slow"

test-gpu:
	$(PY) -m pytest tests -q -m "gpu or slow"

lint:
	$(PY) -m ruff check ways experiments tests scripts docs/images

clean-env:
	PYTHON=$(PY) bash scripts/reproduce_clean_env.sh

figures:
	$(PY) docs/images/src/make_detector_figure.py && $(PY) docs/images/src/make_demand_figure.py
	$(PY) docs/images/render.py

tools:
	$(PY) -m ways.cli tools
