# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# distutils: language = c++
# distutils: sources = qmt/cpp/dummy_src/dummy.cpp
# distutils: include_dirs = qmt/cpp/dummy_src/

import numpy as np
cimport numpy as np

ctypedef np.double_t DOUBLE_t

cdef extern from 'dummy.hpp':
    int c_cppDummy 'cppDummy'()


def cppDummy():
    return c_cppDummy()
