// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <stddef.h>

void quatFromGyrStrapdown(const double* gyr, size_t N, double rate, double* out);

void quatMultiply(const double q1[4], const double q2[4], double out[4]);
void quatConj(const double q[4], double out[4]);
void quatSetToIdentity(double out[4]);
double norm(const double vec[], size_t N);
void normalize(double vec[], size_t N);

#endif // QUATERNION_HPP
