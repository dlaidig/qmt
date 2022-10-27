#!/bin/sh
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# Use this script to run all build steps and tests with one command.

set -e

echo "Recompile extension modules:"
python3 setup.py build_ext --inplace -f
echo ""

echo "Building webapp library:"
./build_webapp_lib.sh
echo ""

echo "Building docs:"
./build_docs.sh
echo ""

echo "Check REUSE compliance:"
reuse lint
echo ""

echo "Checking code style:"
flake8

echo "Running unit tests:"
pytest
