# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# distutils: language = c++
# cython: language_level=3
# cython: embedsignature=True
# distutils: sources = qmt/cpp/madgwick_src/MadgwickAHRS.cpp qmt/cpp/madgwick_src/MadgwickB.cpp qmt/cpp/madgwick_src/MahonyAHRS.cpp
# distutils: include_dirs = qmt/cpp/madgwick_src/
# distutils: undef_macros = NDEBUG

import numpy as np
from libcpp cimport bool
cimport numpy as np
cimport cython

ctypedef np.double_t DOUBLE_t

cdef extern from 'MadgwickAHRS.hpp':
    cdef cppclass C_MadgwickAHRS 'MadgwickAHRS':
        C_MadgwickAHRS(float beta, float sampleFreq) except +

        void update(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz)
        void updateIMU(float gx, float gy, float gz, float ax, float ay, float az)

        float beta
        float q0
        float q1
        float q2
        float q3
        float sampleFreq


cdef class MadgwickAHRS:
    cdef C_MadgwickAHRS* c_obj

    def __cinit__(self, beta, sampleFreq):
        self.c_obj = new C_MadgwickAHRS(<float> beta, <float> sampleFreq)

    def __dealloc__(self):
        del self.c_obj

    def setBeta(self, beta):
        self.c_obj.beta = beta

    def setState(self, np.ndarray[np.double_t, ndim=1, mode='c'] quat not None):
        assert quat.shape[0] == 4

        self.c_obj.q0 = quat[0]
        self.c_obj.q1 = quat[1]
        self.c_obj.q2 = quat[2]
        self.c_obj.q3 = quat[3]

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def update(self, np.ndarray[np.double_t, ndim=1, mode='c'] gyr not None,
               np.ndarray[np.double_t, ndim=1, mode='c'] acc not None,
               np.ndarray[np.double_t, ndim=1, mode='c'] mag not None):
        assert gyr.shape[0] == 3
        assert acc.shape[0] == 3
        assert mag.shape[0] == 3
        self.c_obj.update(<float> gyr[0], <float> gyr[1], <float> gyr[2],
                          <float> acc[0], <float> acc[1], <float> acc[2],
                          <float> mag[0], <float> mag[1], <float> mag[2])

        cdef np.ndarray[double, ndim=1, mode='c'] quat = np.zeros(shape=(4,))
        quat[0] = self.c_obj.q0
        quat[1] = self.c_obj.q1
        quat[2] = self.c_obj.q2
        quat[3] = self.c_obj.q3
        return quat

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def updateIMU(self, np.ndarray[np.double_t, ndim=1, mode='c'] gyr not None,
                  np.ndarray[np.double_t, ndim=1, mode='c'] acc not None):
        assert gyr.shape[0] == 3
        assert acc.shape[0] == 3
        self.c_obj.updateIMU(<float> gyr[0], <float> gyr[1], <float> gyr[2],
                             <float> acc[0], <float> acc[1], <float> acc[2])

        cdef np.ndarray[double, ndim=1, mode='c'] quat = np.zeros(shape=(4,))
        quat[0] = self.c_obj.q0
        quat[1] = self.c_obj.q1
        quat[2] = self.c_obj.q2
        quat[3] = self.c_obj.q3
        return quat

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def updateBatch(self, np.ndarray[np.double_t, ndim=2, mode='c'] gyr not None,
                    np.ndarray[np.double_t, ndim=2, mode='c'] acc not None,
                    np.ndarray[np.double_t, ndim=2, mode='c'] mag=None):
        cdef int N = gyr.shape[0]
        cdef int i = 0
        assert acc.shape[0] == N
        assert gyr.shape[1] == 3
        assert acc.shape[1] == 3

        cdef np.ndarray[double, ndim=2, mode='c'] quat = np.zeros(shape=(N, 4))

        if mag is None:
            for i in range(N):
                self.c_obj.updateIMU(<float> gyr[i, 0], <float> gyr[i, 1], <float> gyr[i, 2],
                                     <float> acc[i, 0], <float> acc[i, 1], <float> acc[i, 2])
                quat[i, 0] = self.c_obj.q0
                quat[i, 1] = self.c_obj.q1
                quat[i, 2] = self.c_obj.q2
                quat[i, 3] = self.c_obj.q3
        else:
            assert mag.shape[0] == N
            assert mag.shape[1] == 3
            for i in range(N):
                self.c_obj.update(<float> gyr[i, 0], <float> gyr[i, 1], <float> gyr[i, 2],
                                  <float> acc[i, 0], <float> acc[i, 1], <float> acc[i, 2],
                                  <float> mag[i, 0], <float> mag[i, 1], <float> mag[i, 2])
                quat[i, 0] = self.c_obj.q0
                quat[i, 1] = self.c_obj.q1
                quat[i, 2] = self.c_obj.q2
                quat[i, 3] = self.c_obj.q3
        return quat


cdef extern from 'MadgwickB.hpp':
    cdef cppclass C_MadgwickB 'MadgwickB':
        C_MadgwickB(float deltat, float beta, float zeta) except +

        void update(float w_x, float w_y, float w_z, float a_x, float a_y, float a_z, float m_x, float m_y, float m_z,
                    bool use_mag)

        float deltat
        float beta
        float zeta

        float SEq_1, SEq_2, SEq_3, SEq_4
        float b_x, b_z
        float w_bx, w_by, w_bz


cdef class MadgwickB:
    cdef C_MadgwickB* c_obj

    def __cinit__(self, deltat, beta, zeta):
        self.c_obj = new C_MadgwickB(<float> deltat, <float> beta, <float> zeta)

    def __dealloc__(self):
        del self.c_obj

    def setState(self, np.ndarray[np.double_t, ndim=1, mode='c'] quat,
                 np.ndarray[np.double_t, ndim=1, mode='c'] bias=None):
        if quat is not None:
            assert quat.shape[0] == 4
            self.c_obj.SEq_1 = quat[0]
            self.c_obj.SEq_2 = quat[1]
            self.c_obj.SEq_3 = quat[2]
            self.c_obj.SEq_4 = quat[3]

        if bias is not None:
            assert bias.shape[0] == 3
            self.c_obj.w_bx = bias[0]
            self.c_obj.w_by = bias[1]
            self.c_obj.w_bz = bias[2]

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def update(self, np.ndarray[np.double_t, ndim=1, mode='c'] gyr not None,
               np.ndarray[np.double_t, ndim=1, mode='c'] acc not None,
               np.ndarray[np.double_t, ndim=1, mode='c'] mag=None):
        assert gyr.shape[0] == 3
        assert acc.shape[0] == 3
        if mag is not None:
            assert mag.shape[0] == 3
            self.c_obj.update(<float> gyr[0], <float> gyr[1], <float> gyr[2],
                              <float> acc[0], <float> acc[1], <float> acc[2],
                              <float> mag[0], <float> mag[1], <float> mag[2], True)
        else:
            self.c_obj.update(<float> gyr[0], <float> gyr[1], <float> gyr[2],
                              <float> acc[0], <float> acc[1], <float> acc[2],
                              0, 0, 0, False)

        cdef np.ndarray[double, ndim=1, mode='c'] quat = np.zeros(shape=(4,))
        quat[0] = self.c_obj.SEq_1
        quat[1] = self.c_obj.SEq_2
        quat[2] = self.c_obj.SEq_3
        quat[3] = self.c_obj.SEq_4
        cdef np.ndarray[double, ndim=1, mode='c'] bias = np.zeros(shape=(3,))
        bias[0] = self.c_obj.w_bx
        bias[1] = self.c_obj.w_by
        bias[2] = self.c_obj.w_bz
        return quat, bias

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def updateBatch(self, np.ndarray[np.double_t, ndim=2, mode='c'] gyr not None,
                    np.ndarray[np.double_t, ndim=2, mode='c'] acc not None,
                    np.ndarray[np.double_t, ndim=2, mode='c'] mag=None):
        cdef int N = gyr.shape[0]
        cdef int i = 0
        assert acc.shape[0] == N
        assert gyr.shape[1] == 3
        assert acc.shape[1] == 3

        cdef np.ndarray[double, ndim=2, mode='c'] quat = np.zeros(shape=(N, 4))
        cdef np.ndarray[double, ndim=2, mode='c'] bias = np.zeros(shape=(N, 3))

        if mag is None:
            for i in range(N):
                self.c_obj.update(<float> gyr[i, 0], <float> gyr[i, 1], <float> gyr[i, 2],
                                  <float> acc[i, 0], <float> acc[i, 1], <float> acc[i, 2],
                                  0, 0, 0, False)
                quat[i, 0] = self.c_obj.SEq_1
                quat[i, 1] = self.c_obj.SEq_2
                quat[i, 2] = self.c_obj.SEq_3
                quat[i, 3] = self.c_obj.SEq_4
                bias[i, 0] = self.c_obj.w_bx
                bias[i, 1] = self.c_obj.w_by
                bias[i, 2] = self.c_obj.w_bz
        else:
            assert mag.shape[0] == N
            assert mag.shape[1] == 3
            for i in range(N):
                self.c_obj.update(<float> gyr[i, 0], <float> gyr[i, 1], <float> gyr[i, 2],
                                  <float> acc[i, 0], <float> acc[i, 1], <float> acc[i, 2],
                                  <float> mag[i, 0], <float> mag[i, 1], <float> mag[i, 2], True)
                quat[i, 0] = self.c_obj.SEq_1
                quat[i, 1] = self.c_obj.SEq_2
                quat[i, 2] = self.c_obj.SEq_3
                quat[i, 3] = self.c_obj.SEq_4
                bias[i, 0] = self.c_obj.w_bx
                bias[i, 1] = self.c_obj.w_by
                bias[i, 2] = self.c_obj.w_bz
        return quat, bias


cdef extern from 'MahonyAHRS.hpp':
    cdef cppclass C_MahonyAHRS 'MahonyAHRS':
        C_MahonyAHRS(float Kp, float Ki, float sampleFreq) except +

        void update(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz)
        void updateIMU(float gx, float gy, float gz, float ax, float ay, float az)

        float twoKp
        float twoKi
        float q0
        float q1
        float q2
        float q3
        float integralFBx
        float integralFBy
        float integralFBz
        float sampleFreq


cdef class MahonyAHRS:
    cdef C_MahonyAHRS* c_obj

    def __cinit__(self, Kp, Ki, sampleFreq):
        self.c_obj = new C_MahonyAHRS(<float> Kp, <float> Ki, <float> sampleFreq)

    def __dealloc__(self):
        del self.c_obj

    def setParams(self, Kp, Ki):
        self.c_obj.twoKp = 2*Kp
        self.c_obj.twoKi = 2*Ki

    def setState(self, np.ndarray[np.double_t, ndim=1, mode='c'] quat,
                 np.ndarray[np.double_t, ndim=1, mode='c'] bias=None):

        if quat is not None:
            assert quat.shape[0] == 4
            self.c_obj.q0 = quat[0]
            self.c_obj.q1 = quat[1]
            self.c_obj.q2 = quat[2]
            self.c_obj.q3 = quat[3]

        if bias is not None:
            assert bias.shape[0] == 3
            self.c_obj.integralFBx = -bias[0]
            self.c_obj.integralFBy = -bias[1]
            self.c_obj.integralFBz = -bias[2]

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def update(self, np.ndarray[np.double_t, ndim=1, mode='c'] gyr not None,
               np.ndarray[np.double_t, ndim=1, mode='c'] acc not None,
               np.ndarray[np.double_t, ndim=1, mode='c'] mag not None):
        assert gyr.shape[0] == 3
        assert acc.shape[0] == 3
        assert mag.shape[0] == 3
        self.c_obj.update(<float> gyr[0], <float> gyr[1], <float> gyr[2],
                          <float> acc[0], <float> acc[1], <float> acc[2],
                          <float> mag[0], <float> mag[1], <float> mag[2])

        cdef np.ndarray[double, ndim=1, mode='c'] quat = np.zeros(shape=(4,))
        quat[0] = self.c_obj.q0
        quat[1] = self.c_obj.q1
        quat[2] = self.c_obj.q2
        quat[3] = self.c_obj.q3
        cdef np.ndarray[double, ndim=1, mode='c'] bias = np.zeros(shape=(3,))
        bias[0] = -self.c_obj.integralFBx
        bias[1] = -self.c_obj.integralFBy
        bias[2] = -self.c_obj.integralFBz
        return quat, bias

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def updateIMU(self, np.ndarray[np.double_t, ndim=1, mode='c'] gyr not None,
                  np.ndarray[np.double_t, ndim=1, mode='c'] acc not None):
        assert gyr.shape[0] == 3
        assert acc.shape[0] == 3
        self.c_obj.updateIMU(<float> gyr[0], <float> gyr[1], <float> gyr[2],
                             <float> acc[0], <float> acc[1], <float> acc[2])

        cdef np.ndarray[double, ndim=1, mode='c'] quat = np.zeros(shape=(4,))
        quat[0] = self.c_obj.q0
        quat[1] = self.c_obj.q1
        quat[2] = self.c_obj.q2
        quat[3] = self.c_obj.q3
        cdef np.ndarray[double, ndim=1, mode='c'] bias = np.zeros(shape=(3,))
        bias[0] = -self.c_obj.integralFBx
        bias[1] = -self.c_obj.integralFBy
        bias[2] = -self.c_obj.integralFBz
        return quat, bias

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def updateBatch(self, np.ndarray[np.double_t, ndim=2, mode='c'] gyr not None,
                    np.ndarray[np.double_t, ndim=2, mode='c'] acc not None,
                    np.ndarray[np.double_t, ndim=2, mode='c'] mag=None):
        cdef int N = gyr.shape[0]
        cdef int i = 0
        assert acc.shape[0] == N
        assert gyr.shape[1] == 3
        assert acc.shape[1] == 3

        cdef np.ndarray[double, ndim=2, mode='c'] quat = np.zeros(shape=(N, 4))
        cdef np.ndarray[double, ndim=2, mode='c'] bias = np.zeros(shape=(N, 3))

        if mag is None:
            for i in range(N):
                self.c_obj.updateIMU(<float> gyr[i, 0], <float> gyr[i, 1], <float> gyr[i, 2],
                                     <float> acc[i, 0], <float> acc[i, 1], <float> acc[i, 2])
                quat[i, 0] = self.c_obj.q0
                quat[i, 1] = self.c_obj.q1
                quat[i, 2] = self.c_obj.q2
                quat[i, 3] = self.c_obj.q3
                bias[i, 0] = -self.c_obj.integralFBx
                bias[i, 1] = -self.c_obj.integralFBy
                bias[i, 2] = -self.c_obj.integralFBz
        else:
            assert mag.shape[0] == N
            assert mag.shape[1] == 3
            for i in range(N):
                self.c_obj.update(<float> gyr[i, 0], <float> gyr[i, 1], <float> gyr[i, 2],
                                  <float> acc[i, 0], <float> acc[i, 1], <float> acc[i, 2],
                                  <float> mag[i, 0], <float> mag[i, 1], <float> mag[i, 2])
                quat[i, 0] = self.c_obj.q0
                quat[i, 1] = self.c_obj.q1
                quat[i, 2] = self.c_obj.q2
                quat[i, 3] = self.c_obj.q3
                bias[i, 0] = -self.c_obj.integralFBx
                bias[i, 1] = -self.c_obj.integralFBy
                bias[i, 2] = -self.c_obj.integralFBz
        return quat, bias
