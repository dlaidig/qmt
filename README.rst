.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

qmt -- IMU motion tracking toolbox
==================================

|tests| |build| |docs| |version| |python| |format| |license| |downloads|

The `qmt` toolbox (Quaternion-based Inertial Motion Tracking Toolbox) is a collection of functions, algorithms,
visualization tools, and other utilities with a focus on IMU-based motion tracking. The `source code
<https://github.com/dlaidig/qmt>`_ is hosted on GitHub.

Documentation
-------------

Detailed documentation can be found online at https://qmt.readthedocs.io/.

Installation
------------

The qmt Python package can easily be installed from `PyPI <https://pypi.org/project/qmt/>`_ via pip, e.g.:

.. code-block:: sh

    pip install qmt

To run webapps, install either `PySide6 <https://pypi.org/project/PySide6/>`_ or
`PySide2 <https://pypi.org/project/PySide2/>`_. If neither PySide6 nor PySide2 work on your system, webapps can still be
displayed with a Chromium browser window fallback.

To install the toolbox from source, run:

.. code-block:: sh

    ./build_webapp_lib.sh
    pip install --user -e ".[dev]"
    ./build_docs.sh

For more information, please refer to the `documentation <https://qmt.readthedocs.io/>`_.

License
-------

The qmt toolbox is licensed under the terms of the `MIT license <https://spdx.org/licenses/MIT.html>`__.

`SPDX <https://spdx.dev/specifications/>`__ headers and the `REUSE <https://reuse.software/>`__ specification are used
to track authors and licenses of all files in this repository.

This repository also contains code, typically released together with scientific publications, for which the original
authors did not provide any licensing information. We distribute this code under the assumption that authors who
publish code for the scientific community intend for the code to be used for scientific research. All such files are
marked with the SPDX license identifier LicenseRef-Unspecified and the origin of the code is documented in the
respective files or directories.

Contact
-------

Daniel Laidig <laidig at control.tu-berlin.de>


.. |tests| image:: https://img.shields.io/github/workflow/status/dlaidig/qmt/Tests?label=tests
    :target: https://github.com/dlaidig/qmt/actions?query=workflow%3ATests
.. |build| image:: https://img.shields.io/github/workflow/status/dlaidig/qmt/Build
    :target: https://github.com/dlaidig/qmt/actions?query=workflow%3ABuild
.. |docs| image:: https://img.shields.io/readthedocs/qmt
    :target: https://qmt.readthedocs.io/
.. |version| image:: https://img.shields.io/pypi/v/qmt
    :target: https://pypi.org/project/qmt/
.. |python| image:: https://img.shields.io/pypi/pyversions/qmt
    :target: https://pypi.org/project/qmt/
.. |format| image:: https://img.shields.io/pypi/format/qmt
    :target: https://pypi.org/project/qmt/
.. |license| image:: https://img.shields.io/pypi/l/qmt
    :target: https://github.com/dlaidig/qmt/blob/main/LICENSES/MIT.txt
.. |downloads| image:: https://img.shields.io/pypi/dm/qmt
    :target: https://pypi.org/project/qmt/
