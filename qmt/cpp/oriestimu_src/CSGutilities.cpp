// SPDX-FileCopyrightText: 2021 Thomas Seel <seel@control.tu-berlin.de>
// SPDX-FileCopyrightText: 2021 Stefan Ruppin <ruppin@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

//! IMU ORIENTATION ESTIMATION
//! Developed at Control Systems Group, TU Berlin
//! (http://www.control.tu-berlin.de)
//! Algorithm design: Thomas Seel
//! Code design: Stefan Ruppin
//! Further reading: paper to be submitted...
//! Contact: {seel,ruppin}@control.tu-berlin.de

#include <math.h>
#include <stdio.h>
#include "CSGutilities.hpp"



void quaternionMultiply(double q1[R4], double q2[R4], double qResult[R4])
{
    qResult[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    qResult[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    qResult[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    qResult[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
}



void quaternionInvert(double q[R4], double qInverted[R4])
{
    if (q!= NULL && qInverted != NULL)
    {
        qInverted[0] =  q[0];
        qInverted[1] = -q[1];
        qInverted[2] = -q[2];
        qInverted[3] = -q[3];
    }
}



void quaternionCoordTransform(double q1[R4], double q2[R4], double qResult[R4])
{
	double qInverted[R4];
	double qMult[R4];

	quaternionInvert(q1, qInverted);
	quaternionMultiply(qInverted, q2, qMult);
	quaternionMultiply(qMult, q1, qResult);
}


double norm(double vec[], int length)
{
	int i;
	double result = 0;
	for (i=0; i < length; i++) {
		result += pow(vec[i],2);
	}

	return sqrt(result);
}

void normalize (double vec[], int length, double vecNorm,
                double normalizedResult[])
{
    int i;
    if (vecNorm != 0) {
        for (i=0; i<length; i++) {
            normalizedResult[i] = vec[i] / vecNorm;
        }
    } else {
        for (i=0; i<length; i++)
            normalizedResult[i] = 0;
    }
}

void vectorCrossProduct(double a[R3], double b[R3], double result[R3])
{
	result[0] = a[1] * b[2] - a[2] * b[1];
	result[1] = a[2] * b[0] - a[0] * b[2];
	result[2] = a[0] * b[1] - a[1] * b[0];
}


double scalarProduct(double a[R3], double b[R3])
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
