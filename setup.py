"""
Install:
    pip3 install --user -e ".[dev]"

Recompile extension modules
    python3 setup.py build_ext --inplace -f
"""
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT


# https://stackoverflow.com/a/60740179
# (note that even with pyproject.toml this is still useful to make `python setup.py sdist` work out-of-the-box)
from setuptools import dist
dist.Distribution().fetch_build_eggs(['Cython', 'numpy'])

import site
import sys
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

# workaround for develop mode (pip install -e) with PEP517/pyproject.toml cf. https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = '--user' in sys.argv[1:]

ext_modules = cythonize([
    'qmt/cpp/oriestimu.pyx',
    'qmt/cpp/madgwick.pyx',
    'qmt/cpp/quaternion.pyx',
])

for m in ext_modules:
    m.include_dirs.insert(0, np.get_include())

setup(
    name='qmt',
    version='0.2.1',

    description='Quaternion-based Inertial Motion Tracking Toolbox',
    long_description=open('README.rst', encoding='utf-8').read(),
    long_description_content_type="text/x-rst",
    url='https://github.com/dlaidig/qmt/',
    project_urls={
        'Documentation': 'https://qmt.readthedocs.io/',
    },

    author='Daniel Laidig',
    author_email='laidig@control.tu-berlin.de',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    include_package_data=True,
    zip_safe=False,

    install_requires=['numpy', 'scipy', 'matplotlib', 'PyYAML',
                      'transplant>=0.8.11',  # 0.8.11 fixes https://github.com/bastibe/transplant/issues/81
                      'aiohttp>=3.8.1', 'aiofiles', 'orjson', 'qasync', 'vqf'],
    extras_require={
        # pip3 install --user -e ".[dev]"
        'dev': ['pytest', 'pytest-flake8', 'flake8<4',  # https://github.com/tholo/pytest-flake8/issues/81
                'reuse', 'Cython', 'sphinx', 'recommonmark', 'sphinx-rtd-theme', 'sphinxcontrib-matlabdomain'],
    },

    entry_points={
        'console_scripts': [
            'qmt-webapp = qmt.utils.webapp_cli:main',
        ],
    },

    ext_modules=ext_modules,

    include_dirs=[np.get_include()],
)
