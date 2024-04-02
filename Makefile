#!/usr/bin/env python

# -*- coding: utf-8 -*-

#
# Copyright (c) 2023-2024, Cyrille Favreau (cyrille.favreau@gmail.com)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#see if we're in a virtualenv, and use that, otherwise use the default
ifdef VIRTUAL_ENV
   VENV=$(VIRTUAL_ENV)
else
   VENV:=venv
endif
VENV_BIN:=$(VENV)/bin

# simulate running in headless mode
unexport DISPLAY

# Test coverage pass threshold (percent)
MIN_COV?=45
VENV_INSTALLED=.installed
PIP_INSTALL_OPTIONS=--ignore-installed --no-deps

FIND_LINT_PY=`find nuon_model_visualizer -name "*.py" -not -path "*/*test*"`
LINT_PYFILES := $(shell find $(FIND_LINT_PY))

$(VENV):
	virtualenv --system-site-packages $(VENV)

$(VENV_INSTALLED): $(VENV)
	$(VENV_BIN)/pip install $(PIP_INSTALL_OPTIONS) -r requirements_dev.txt
	$(VENV_BIN)/pip install -e .
	touch $@

run_pycodestyle: $(VENV_INSTALLED)
	$(VENV_BIN)/pycodestyle $(LINT_PYFILES) > pycodestyle.txt

run_pydocstyle: $(VENV_INSTALLED)
	$(VENV_BIN)/pydocstyle $(LINT_PYFILES) > pydocstyle.txt

run_pylint: $(VENV_INSTALLED)
	$(VENV_BIN)/pylint --rcfile=pylintrc $(LINT_PYFILES) > pylint.txt

run_tests: $(VENV_INSTALLED)
	$(VENV_BIN)/nosetests -v --with-coverage --cover-min-percentage=$(MIN_COV) --cover-erase --cover-package nuon_model_visualizer

run_tests_xunit: $(VENV_INSTALLED)
	@mkdir -p $(ROOT_DIR)/test-reports
	$(VENV_BIN)/nosetests nuon_model_visualizer --with-coverage --cover-min-percentage=$(MIN_COV) --cover-inclusive --cover-erase --cover-package=nuon_model_visualizer --with-xunit --xunit-file=nosetests_nuon_model_visualizer.xml

lint: run_pycodestyle run_pydocstyle run_pylint

test: lint run_tests

doc: $(VENV_INSTALLED)
	make SPHINXBUILD=$(VENV_BIN)/sphinx-build -C doc html

doc_pdf: $(VENV_INSTALLED)
	make SPHINXBUILD=$(VENV_BIN)/sphinx-build -C doc latexpdf

clean_test_venv:
	@rm -rf $(VENV_INSTALLED)
	@rm -rf $(ROOT_DIR)/test-reports

clean_doc:
	@test -x $(VENV_BIN)/sphinx-build && make SPHINXBUILD=$(VENV_BIN)/sphinx-build  -C doc clean || true
	@rm -rf $(ROOT_DIR)/doc/build

clean: clean_doc clean_test_venv
	@rm -f pycodestyle.txt
	@rm -f pydocstyle.txt
	@rm -f pylint.txt
	@rm -rf nuon_model_visualizer/nuon_model_visualizer.egg-info
	@rm -f .coverage
	@rm -rf test-reports
	@rm -rf dist
	@rm -f $(VENV_INSTALLED)

.PHONY: run_pycodestyle test clean_test_venv clean doc
