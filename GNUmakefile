.DEFAULT_GOAL := all

.PHONY: all release release-env

PYTHON ?= .venv314/bin/python
UV_CACHE_DIR ?= .uv-cache
RELEASE_ARGS ?=

all:
	$(MAKE) -f Makefile all

release: release-env
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) scripts/release.py $(RELEASE_ARGS)

release-env:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv venv --python 3.14 .venv314
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv pip install --python $(PYTHON) "build==1.5.0" "cibuildwheel==3.3.1" "cmake==4.3.2" "delocate==0.13.0" "setuptools==81.0.0" "twine==6.2.0" "wheel==0.47.0"

%:
	$(MAKE) -f Makefile $@
