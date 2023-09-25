SHELL = /bin/bash

.DEFAULT_GOAL := all

## help: Display list of commands
.PHONY: help
help: Makefile
	@sed -n 's|^##||p' $< | column -t -s ':' | sed -e 's|^| |'

## all: Run all targets
.PHONY: all
all: init style test

## init: Bootstrap dev env.
.PHONY: init
init:
	python3 -m venv ./venv
	./venv/bin/pip install -e .[tests]

## test: Shortcut to launch all the test on all environments
.PHONY: test
test:
	./venv/bin/pytest

## test-all: Shortcut to launch all the test on all environments
.PHONY: test-all
test-all:
	./venv/bin/tox p

## clean: Remove temporary files
.PHONY: clean
clean:
	-rm -rf .mypy_cache ./**/.pytest_cache venv
