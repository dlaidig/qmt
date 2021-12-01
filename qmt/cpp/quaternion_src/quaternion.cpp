// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

#include "quaternion.hpp"

#include <limits>
#include <algorithm>
#include <math.h>

void quatFromGyrStrapdown(const double *gyr, size_t N, double rate, double *out)
{
    quatSetToIdentity(out);
    double q[4] = {1, 0, 0, 0};

    for (size_t i = 0; i < N; i++) {
        double gyrnorm = norm(gyr+3*i, 3);
        double angle = gyrnorm / rate;
        if (gyrnorm > std::numeric_limits<double>::epsilon()) {
            double c = cos(angle/2);
            double s = sin(angle/2);
            double gyrStepQuat[4] = {c, s*gyr[3*i]/gyrnorm, s*gyr[3*i+1]/gyrnorm, s*gyr[3*i+2]/gyrnorm};
            quatMultiply(q, gyrStepQuat, q);
            normalize(q, 4);
        }
        std::copy(q, q+4, out+4*i);
    }
}

void quatMultiply(const double q1[4], const double q2[4], double out[4])
{
    double w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    double x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    double y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    double z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

void quatConj(const double q[4], double out[4])
{
    double w = q[0];
    double x = -q[1];
    double y = -q[2];
    double z = -q[3];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

void quatSetToIdentity(double out[4])
{
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
}

double norm(const double vec[], size_t N)
{
    double s = 0;
    for(size_t i = 0; i < N; i++) {
        s += vec[i]*vec[i];
    }
    return sqrt(s);
}

void normalize(double vec[], size_t N)
{
    double n = norm(vec, N);
    if (n < std::numeric_limits<double>::epsilon()) {
        return;
    }
    for(size_t i = 0; i < N; i++) {
        vec[i] /= n;
    }
}
