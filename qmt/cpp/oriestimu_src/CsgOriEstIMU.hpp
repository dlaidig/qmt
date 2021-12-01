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

//! How to use the filter:
//! Frist, call the Constructor of the class.
//! There, specify the sampling frequency, a start quaternion[4] and an
//! initial bias[3] to begin the estimation with. Choose the quaternion unit.
//! If you want to reset the filter and to change the inital quaternion or bias,
//! call the member function CCsgOriEstIMU::resetEstimation().
//! Additionally, useMeasRawRatingIn can be set in the Constructor.
//! If useMeasRawRatingIn == 0, no rating of the raw accelerometer data is used.
//! If useMeasRawRatingIn > 0, the raw accelerometer data is rated and the
//! trust will be reduced for high accelerometer norm
//! Recommended: useMeasRawRatingIn = 1
//!
//! After calling the Constructor, call the member function:
//! CCsgOriEstIMU::setFilterParameter()
//! This function will set the filter parameter.
//! Choose for TauAcc the time that should pass in seconds before the error of
//! the inclination is halved.
//! Analog for TauMag.
//! Choose for Zeta a value >= 0. This value will define your overshooting.
//! Zeta = 0 means no bias estimation will be done.
//! Zeta > 0 means that a bias will be estimated.
//! The higher Zeta is chosen, the more overshooting will be present
//! when the reference jumps and the faster the bias will be estimated.
//! Recommended: Zeta = 1
//!
//! Now, that the filter in initialized, the member function
//! CCsgOriEstIMU::updateCsgOriEstIMU can be called.
//! This function needs the current measurements (or at least the memory of it)
//! The function will generate three outputs:
//! 	the updated estimation of orientation and bias as well as the errors.
//! Provide memory for the output in the three last elements.
//! Should one of the argument pointer be NULL, the updated will be aborted.
//! If the update was successful, the function will return SUCCESS, otherwise
//! it will return ERROR.
//!
//! At the beginning of the estimation the algoriithm will use a time dependend
//! small time constant for orientation estimation to ensure fast convergence
//! after start of the algorithm.


#ifndef __CCsgOriEstIMU__
	#define __CCsgOriEstIMU__

	#include "CSGutilities.hpp"
	#include <stdint.h>

	static const int8_t ERROR = -1;
  	static const int8_t SUCCESS = 0;
	static const int8_t  Incl = 0;
	static const int8_t Azi  = 1;

    //! Class: CsgOriEstIMU
    //! Purpose of this class is to implement functions and define the variables
    //! needed to perform an orientation estimation based on inertial sensor
    //! data (accelerometer, magnetometer, gyro).
	class CCsgOriEstIMU {
		private:
            //! correction factor for accelerometer correction
			double correctionAccGain;
            //! correction factor for magnetometer correction
		   	double correctionMagGain;
            //! factor for bias accumulation calculated from user input
			double correctionBiasGain;


            //! sample frequency of the input data
			double sampleFreq;

            //! init for fixed gravitation direction    (as quaternion)
			double gravrefFixedframe[R4];
			//! define magnetic field reference in fixed frame
			//! (choose [+/-1 0 0] or [0 +/-1 0] to define which fixed-frame coordinate axis points north/south)
            double magrefFixedFrame[R4];
            //! current estimation of fixed gravitation in imu frame (quaternion)
			double gravrefImuframe[R4];
			//! current estimation of fixed magnetic field in imu frame (quaternion)
            double magrefImuframe[R4];

            //! internal state of estimation
		    double storedBias[R3];
		    double storedQuaternion[R4];

            //!
            const int8_t windowLength;
            //!
            double *ratingWindow;
            //!
            double useMeasRawRating;
			//!
			double validAccDataCount;
		public:
			//! saves tauAcc (time constant)
			double tauAcc;
			//! save tauMag (time constant)
			double tauMag;
			//! saves zeta (integral action modifier)
			double zeta;

            //! Constructor for CsgOriEstIMU Algorithm
            //! Will initialize the start/initial values for the quaternion and the bias
            //! as well as set the used sampling frequency
            //!
            //! @param sampleFreq sampling frequncy of measured data
            //! @param initQuaternion initial orientation to begin estimation with
            //! @param initBias inital bias to begin bias estimation with
		    CCsgOriEstIMU (double sampleFreq, double initQuaternion[R4],
                           double initGyroBias[R3], double useMeasRawRating);


            //! Destructor
            ~CCsgOriEstIMU();


            //!  This function will do one update step for the orientation estimation.
            //!  Its start orientation is saved as state in the class
            //!  The new measurements will be used to update this orientation.
            //!  The result will be saved as the new state and will returned.
            //!  Additionally a bias correction on the gyro will be performed.
			//!  Uses SI units.
			//!
			//!  @param accmeas  array of accelerometer data (dim: 3) [m/s^2]
			//!  @param gyromeas array of gyro data (dim: 3)          [rad/s]
			//!  @param magmeas  array of magnetometer data (dim: 3)  [any unit]
            //!
			//!  @param outputQuaternion    estimated orientation after update
			//!  @param outputEstimatedBias estimated bias after update
		    int8_t updateCsgOriEstIMU 	 (double accmeas[R3],
                                          double gyromeas[R3],
                                          double magmeas[R3],
		                                  double outputQuaternion[R4],
                                          double outputEstimatedBias[R3],
                                          double outputErrors[2]);

            //! This function is a setter function to set the filter parameter.
            //!
            //! @param tauAcc  half-life(50%) for the error determined between
            //!                gyro prediction and accelerometer data
            //!                note: in seconds
            //! @param tauMag  half-life(50%) for the error determined between
            //!                accelerometer corrected orientation and magnetometer data
            //!                note: in seconds
            //! @param zeta    determines the overshooting/undershooting behaviour by influencing
            //!                the bias calculation
            //!                <= 1: almost no overshooting
            //!                >  1: overshooting is getting bigger
		    int8_t setFilterParameter(double tauAcc, double tauMag, double zeta);


			//! This function calculates the correction gains out of the time
			//! constants for the error correction.
			//! @param tauAcc  half-life(50%) time for accelerometer correction
            //!                note: in seconds
            //! @param tauAcc  half-life(50%) time for magnetometer correction
            //!                note: in seconds
            //! @param zeta    factor for bias correction
			//! @param outputCorrectionAccGain 	calculated correction gain for
			//!								   	accelerometer correction
			//! @param outputCorrectionMagGain  calculated correction gain for
			//! 								magnetometer correction
			//! @param outputCorrectionBiasGain calculated correction gain for
			//!									bias estimation
			int8_t calculateCorrectionGains(double tauAcc, double tauMag,
											double zeta);

            //! This function is used to reset the interal estimation state of the
            //! filter function.
            //! This means that this can reset the filter and set new starting orientation
            //! and bias for estimation.
            //! @param newQuaternion new start values of orientation in quaternion (dim 4)
            //!  @param newBias      new start values for bias estimation (dim 3)
            void resetEstimation (double newQuaternion[R4], double newBias[R3]);

            void setRating(double rating); // HACK BY DL
	};

#endif
