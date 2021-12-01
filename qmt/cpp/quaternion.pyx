# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# distutils: language = c++
# cython: language_level=3
# distutils: undef_macros = NDEBUG
# distutils: sources = qmt/cpp/quaternion_src/quaternion.cpp

import numpy as np
cimport numpy as np


cdef extern from 'quaternion_src/quaternion.hpp':
    cdef void c_quatFromGyrStrapdown 'quatFromGyrStrapdown'(const double* gyr, size_t N, double rate, double* out)


def quatFromGyrStrapdown(np.ndarray[np.double_t, ndim=2, mode="c"] gyr not None, double rate):
    cdef int N = gyr.shape[0]
    assert gyr.shape[1] == 3

    cdef np.ndarray[double, ndim=2, mode="c"] out = np.zeros(shape=(N, 4))

    c_quatFromGyrStrapdown(<double*> np.PyArray_DATA(gyr), N, rate, <double*> np.PyArray_DATA(out))
    return out
