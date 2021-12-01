# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import os
import pytest
import logging

import qmt


def pytest_addoption(parser):
    parser.addoption('--nomatlab', action='store_true', default=False,
                     help='disable tests that need Matlab and transplant')


def pytest_configure(config):
    config.addinivalue_line('markers', 'matlab: mark test that need Matlab and transplant')

    # silence warnings regarding argparse use from pytest-flake8, cf. https://github.com/tholo/pytest-flake8/issues/69
    logging.getLogger('flake8').setLevel(logging.ERROR)


def pytest_collection_modifyitems(config, items):
    if config.getoption('--nomatlab'):
        skipMatlab = pytest.mark.skip(reason='disabled via --nomatlab option')
        for item in items:
            if 'matlab' in item.keywords:
                item.add_marker(skipMatlab)


@pytest.fixture(scope='session')
def example_data():
    data = qmt.Struct.load(os.path.join(os.path.dirname(__file__), '..', 'examples', 'full_body_example_data.mat'))
    data['rate'] = 100
    return data
