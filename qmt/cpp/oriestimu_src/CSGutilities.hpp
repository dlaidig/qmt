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

#ifndef __CSGUTILITIES__
	#define __CSGUTILITIES__

    #include <stdint.h>

	static const int8_t R3 = 3;
	static const int8_t R4 = 4;


    //!  This will multiply two quaternions qResult = q1*q2
	void quaternionMultiply(double q1[R4], double q2[R4], double qResult[R4]);


    //! This function will invert a given quaternion
    //! qInverted = {q[0] -q[1] - q[2] -q[3]}
	void quaternionInvert(double q[R4], double qInverted[R4]);


    //! This function will compute a coordinate transformation
    //! qResult = q1' * q2 * q1
	void quaternionCoordTransform(double q1[R4], double q2[R4], double qResult[R4]);


    //! This function will calculate the norm of an vector of dimension dim
	double norm(double vec[], int length);


    //! Normalize a vector (after: norm(vector)=1)
	void normalize(double vec[], int length, double vecNorm, double normalizedResult[]);


    //! Calculate the vector product of a and b
	void vectorCrossProduct(double a[R3], double b[R3], double result[R3]);


    //! Standard scalar product: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
	double scalarProduct(double a[R3], double b[R3]);
#endif
