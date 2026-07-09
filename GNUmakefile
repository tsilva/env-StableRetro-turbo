.DEFAULT_GOAL := all

.PHONY: all benchmark benchmark-build benchmark-local release release-env

PYTHON ?= .venv314/bin/python
UV_CACHE_DIR ?= .uv-cache
RELEASE_ARGS ?=
BENCHMARK_BUILD_PYTHON ?= .venv314/bin/python
BENCHMARK_PYTHON ?= $(BENCHMARK_BUILD_PYTHON)
BENCHMARK_BUILD_ENV ?= STABLE_RETRO_BUILD_ROSETTA_SNES=0
BENCHMARK_EXT_SUFFIX ?= $(shell $(BENCHMARK_BUILD_PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
BENCHMARK_EXTENSION ?= stable_retro/_retro$(BENCHMARK_EXT_SUFFIX)
BENCHMARK_BUILD_INPUTS := CMakeLists.txt setup.py $(wildcard src/*.cpp src/*.h stable_retro/*.py stable_retro/VERSION.txt)
BENCHMARK_PROFILE ?= supermario-level1-1
BENCHMARK_BACKEND ?= auto
BENCHMARK_SECONDS ?= 10
BENCHMARK_WARMUP_STEPS ?= 16
GAME ?=
PLATFORM ?=
STATE ?=
BENCHMARK_GAME ?= $(GAME)
BENCHMARK_PLATFORM ?= $(PLATFORM)
BENCHMARK_STATE ?= $(STATE)
BENCHMARK_ARGS ?=

all:
	$(MAKE) -f Makefile all

benchmark: $(BENCHMARK_EXTENSION) benchmark-local

benchmark-build: $(BENCHMARK_EXTENSION)

$(BENCHMARK_EXTENSION): $(BENCHMARK_BUILD_INPUTS)
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(BENCHMARK_BUILD_ENV) $(BENCHMARK_BUILD_PYTHON) setup.py build_ext --inplace
	test -f $@

benchmark-local:
	PYTHONPATH=$(CURDIR) $(BENCHMARK_PYTHON) scripts/benchmark_vec_env.py --profile $(BENCHMARK_PROFILE) --backend $(BENCHMARK_BACKEND) --seconds $(BENCHMARK_SECONDS) --warmup-steps $(BENCHMARK_WARMUP_STEPS) $(if $(BENCHMARK_GAME),--game $(BENCHMARK_GAME)) $(if $(BENCHMARK_PLATFORM),--platform $(BENCHMARK_PLATFORM)) $(if $(BENCHMARK_STATE),--state $(BENCHMARK_STATE)) $(BENCHMARK_ARGS)

release: release-env
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(PYTHON) scripts/release.py $(RELEASE_ARGS)

release-env:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv venv --allow-existing --python 3.14 .venv314
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv pip install --python $(PYTHON) "build==1.5.0" "cibuildwheel==3.3.1" "cmake==4.3.2" "delocate==0.13.0" "farama-notifications==0.0.6" "gymnasium==1.2.3" "pyglet==1.5.31" "setuptools==81.0.0" "twine==6.2.0" "wheel==0.47.0"

%:
	$(MAKE) -f Makefile $@
