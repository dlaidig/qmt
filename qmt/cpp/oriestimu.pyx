# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# distutils: language = c++
# cython: language_level=3
# cython: embedsignature=True
# distutils: sources = qmt/cpp/oriestimu_src/CsgOriEstIMU.cpp qmt/cpp/oriestimu_src/CSGutilities.cpp
# distutils: include_dirs = qmt/cpp/oriestimu_src/

import numpy as np
cimport numpy as np

ctypedef np.double_t DOUBLE_t

cdef extern from 'CsgOriEstIMU.hpp':
    cdef cppclass CCsgOriEstIMU:
        CCsgOriEstIMU(double, double[4], double[3], double) except +
        signed char setFilterParameter(double, double, double)
        signed char updateCsgOriEstIMU(double[3], double[3], double[3], double*, double*, double*)
        void resetEstimation(double[4], double[3])
        void setRating(double rating)


cdef class OriEstIMU:
    cdef CCsgOriEstIMU* c_oriEstImu
    cdef public object quat
    cdef public object bias
    cdef public object error

    def __cinit__(self, double rate, double tauAcc, double tauMag, double zeta, double accRating):
        self.c_oriEstImu = new CCsgOriEstIMU(rate, [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0], accRating)
        self.c_oriEstImu.setFilterParameter(tauAcc, tauMag, zeta)
        self.quat = None
        self.bias = None
        self.error = None

    def __dealloc__(self):
        del self.c_oriEstImu

    def update(self, np.ndarray[DOUBLE_t, ndim=1, mode='c'] acc not None,
               np.ndarray[DOUBLE_t, ndim=1, mode='c'] gyr not None,
               np.ndarray[DOUBLE_t, ndim=1, mode='c'] mag not None):
        if acc.shape[0] != 3 or gyr.shape[0] != 3 or mag.shape[0] != 3:
            raise ValueError('acc, gyr and mag must have length 3')
        cdef double[4] quat
        cdef double[3] bias
        cdef double[2] error
        self.c_oriEstImu.updateCsgOriEstIMU(<double*> np.PyArray_DATA(acc), <double*> np.PyArray_DATA(gyr),
                                            <double*> np.PyArray_DATA(mag), quat, bias, error)
        self.quat = quat
        self.bias = bias
        self.error = error
        return quat, bias, error

    def updateBatch(self, np.ndarray[np.double_t, ndim=2, mode='c'] acc not None,
                    np.ndarray[np.double_t, ndim=2, mode='c'] gyr not None,
                    np.ndarray[np.double_t, ndim=2, mode='c'] mag=None):
        cdef int N = gyr.shape[0]
        cdef int i = 0
        assert acc.shape[0] == N
        assert gyr.shape[1] == 3
        assert acc.shape[1] == 3

        cdef np.ndarray[double, ndim=1, mode='c'] zeromag = np.zeros(shape=(3,))

        cdef np.ndarray[double, ndim=2, mode='c'] quat = np.zeros(shape=(N, 4))
        cdef np.ndarray[double, ndim=2, mode='c'] bias = np.zeros(shape=(N, 3))
        cdef np.ndarray[double, ndim=2, mode='c'] error = np.zeros(shape=(N, 2))

        if mag is None:
            for i in range(N):
                self.c_oriEstImu.updateCsgOriEstIMU((<double*> np.PyArray_DATA(acc))+3*i,
                                                    (<double*> np.PyArray_DATA(gyr))+3*i,
                                                    <double*> np.PyArray_DATA(zeromag),
                                                    (<double*> np.PyArray_DATA(quat))+4*i,
                                                    (<double*> np.PyArray_DATA(bias))+3*i,
                                                    (<double*> np.PyArray_DATA(error))+2*i)
        else:
            assert mag.shape[0] == N
            assert mag.shape[1] == 3
            for i in range(N):
                self.c_oriEstImu.updateCsgOriEstIMU((<double*> np.PyArray_DATA(acc))+3*i,
                                                    (<double*> np.PyArray_DATA(gyr))+3*i,
                                                    (<double*> np.PyArray_DATA(mag))+3*i,
                                                    (<double*> np.PyArray_DATA(quat))+4*i,
                                                    (<double*> np.PyArray_DATA(bias))+3*i,
                                                    (<double*> np.PyArray_DATA(error))+2*i)
        return quat, bias, error

    def resetEstimation(self, np.ndarray[DOUBLE_t, ndim=1, mode='c'] newQuat not None,
                        np.ndarray[DOUBLE_t, ndim=1, mode='c'] newBias not None):
        if newQuat.shape[0] != 4 or newBias.shape[0] != 3:
            raise ValueError('newQuat must have length 4 and newBias must have length 3')
        self.c_oriEstImu.resetEstimation(<double*> np.PyArray_DATA(newQuat), <double*> np.PyArray_DATA(newBias))

    def setFilterParameter(self, double tauAcc, double tauMag, double zeta):
        self.c_oriEstImu.setFilterParameter(tauAcc, tauMag, zeta)

    def setRating(self, double rating):
        self.c_oriEstImu.setRating(rating)
