.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

Introduction
############

The `qmt` toolbox (Quaternion-based Inertial Motion Tracking Toolbox) is a collection of functions, algorithms,
visualization tools, and other utilities with a focus on IMU-based motion tracking.

The main goal is to provide high-level access to various algorithms in a consistent way that makes it possible to
analyze new experimental data with just a few lines of code.

Installation
------------

The qmt Python package can easily be installed from `PyPI <https://pypi.org/project/qmt/>`_ via pip, e.g.:

.. code-block:: sh

    pip install qmt

Depending on your Python installation, it might be necessary to use ``pip3`` instead of ``pip`` and/or to add the
``--user`` option.
To also install the `PySide2 <https://pypi.org/project/PySide2/>`_ package that is needed to run webapps in a custom
window, use the extra ``gui`` specifier: ``pip install "qmt[gui]"``. If installing PySide2 via PyPI is not supported for
your architecture, you can either find other ways to install it (e.g.,
`brew <https://formulae.brew.sh/formula/pyside@2>`_) or use the Chromium browser window fallback.

To install the toolbox from `the source code hosted on GitHub <https://github.com/dlaidig/qmt>`_, the JavaScript library
for the webapps needs to be built first. It is also recommended to install the optional ``dev`` dependencies. In the
directory of the git repository, run the following commands:

.. code-block:: sh

    ./build_webapp_lib.sh
    pip3 install --user -e ".[dev]"

To build the documentation locally, run ``./build_docs.sh``. To view the documentation, open ``documentation.html`` in a
browser.
