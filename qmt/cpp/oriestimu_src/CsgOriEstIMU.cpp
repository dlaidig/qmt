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

// for memmove
#include <string.h>
// for trig function
#include <math.h>
// for printf
#include <stdio.h>
// for calloc
#include <stdlib.h>
// for helo functions like quaternionMultiply(), norm(), normalize()
#include "CSGutilities.hpp"
// Class headder
#include "CsgOriEstIMU.hpp"


//! Constructor for CsgOriEstIMU Algorithm
//! Will initialize the start/initial values for the quaternion and the bias
//! as well as set the used sampling frequency
//!
//! @param sampleFreq sampling frequncy of measured data
//! @param initQuaternion initial orientation to begin estimation with
//! @param initBias inital bias to begin bias estimation with
//! @param useMeasRawRatingIn 0: do not use raw rating of accelerometer
//!                           1: use it
//!                           recommended: 1
CCsgOriEstIMU::CCsgOriEstIMU (double sampleFreq,
                              double initQuaternion[R4],
                              double initBias[R3],
                              double useMeasRawRatingIn)
:windowLength(10)
{
    // Sets private class variables
	this->sampleFreq = sampleFreq;
    if (initQuaternion != NULL && initBias != NULL)
      {
        normalize(initQuaternion, R4, norm(initQuaternion, R4),             // normalize input quaternion to be safe
                    this->storedQuaternion);
        memmove(this->storedBias, initBias, sizeof(double)*R3);
      }

	// define and set the fixed frames
	this->gravrefFixedframe[0] = 0.0;
	this->gravrefFixedframe[1] = 0.0;
	this->gravrefFixedframe[2] = 0.0;
	this->gravrefFixedframe[3] = 1.0;

	this->magrefFixedFrame[0] = 0.0;
	this->magrefFixedFrame[1] = 0.0;
	this->magrefFixedFrame[2] = 1.0;
	this->magrefFixedFrame[3] = 0.0;

    // set default filter values in case setFilterParameter was not called
    this->tauAcc = 1.0;
    this->tauMag = 1.0;
    this->zeta = 1.0;
    calculateCorrectionGains(this->tauAcc, this->tauMag, this->zeta);

    // set call count to zero
    this->validAccDataCount = 0;

    // get memory for the window to use it if needed!
    // if (0 < useMeasRawRatingIn) // CHANGED BY DL
    // {
        this->ratingWindow = (double *)
                            calloc(this->windowLength,sizeof(double));
    // }
    this->useMeasRawRating = useMeasRawRatingIn;

} // CsgOriEstIMU::CsgOriEstIMU


CCsgOriEstIMU::~CCsgOriEstIMU(void)
{
    // if (0 < this->useMeasRawRating) // CHANGED BY DL
    //  {
        free(this->ratingWindow);
    //  }
}


//! This function is used to reset the interal estimation state of the
//! filter function.
//! This means that this can reset the filter and set new starting orientation
//! and bias for estimation.
//! @param newQuaternion new start values of orientation in quaternion (dim 4)
//! @param newBias       new start values for bias estimation (dim 3)
void CCsgOriEstIMU::resetEstimation(double newQuaternion[R4], double newBias[R3])
{
    if (newQuaternion != NULL && newBias != NULL)
      {
        normalize(newQuaternion, R4, norm(newQuaternion, R4),               // normalize input quaternion to be safe
                      this->storedQuaternion);
        memmove(this->storedBias, newBias, sizeof(double)*R3);
      }
} // CsgOriEstIMU::resetEstimation


//! This function is a setter function to set the filter parameter.
//!
//! @param tauAcc  half-life(50%) for the error determined between
//!                gyro prediction and accelerometer data
//!                note: in seconds
//! @param tauAcc  half-life(50%) for the error determined between
//!                accelerometer corrected orientation and magnetometer data
//!                note: in seconds
//! @param zeta    determines the overshooting/undershooting behaviour by influencing
//!                the bias calculation
//!                == 0: no bias estimation
//!                <= 1: almost no overshooting
//!                >  1: overshooting is getting bigger
//!                Recommended: zeta = 1
int8_t CCsgOriEstIMU::setFilterParameter(double tauAcc, double tauMag, double zeta)
{
    if (0 == tauAcc || 0 == tauMag)
     {
        printf("Error while setting filter parameter: either tauMag or tauAcc is zero \n\n");
        return ERROR;
     }

    calculateCorrectionGains(tauAcc,tauMag,zeta);

    this->tauAcc = tauAcc;
    this->tauMag = tauMag;
    this->zeta   = zeta;

    return SUCCESS;
} // CsgOriEstIMU::setFilterParameter


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
int8_t CCsgOriEstIMU::calculateCorrectionGains(double tauAcc, double tauMag,
                                double zeta)
{
    this->correctionAccGain  = 1.0 - 1.4 * tauAcc * sampleFreq
                                 / (1.4 * tauAcc * sampleFreq + 1);
    this->correctionMagGain  = 1.0 - 1.4 * tauMag * sampleFreq
                                 / (1.4 * tauMag * sampleFreq + 1);
    this->correctionBiasGain = zeta * zeta / 160.0 * 1.4 * this->sampleFreq;

    return 0;
}


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
//   @param outputErrors 1:Error of Inclination, 2: error of azimuth
int8_t CCsgOriEstIMU::updateCsgOriEstIMU (double accmeas[R3],
                                          double gyromeas[R3],
                                          double magmeas[R3],
                                          double outputQuaternion[R4],
                                          double outputEstimatedBias[R3],
                                          double outputErrors[2])
{
    // check if inputs and outputs are valid memory
    if ( accmeas == NULL || gyromeas == NULL || magmeas == NULL
         || outputQuaternion == NULL || outputEstimatedBias == NULL)
      {
        printf("Error during filer update: NULL pointer in argument \n \n");
        return ERROR;
      }

    // set default output for errors
    outputErrors[0] = 0;
    outputErrors[1] = 0;

	double gyromeasBiasCorrected[R3];

	double normedCorrectionaxisImuframe[R3];
	double normedQMagAccGyro[R4];

    double predictionAngle;
	double correctionAngle;
	double correctionaxisImuframe[R3];

	double qGyro[R4];
	double qGyroAcc[R4];
	double qMagAccGyro[R4];

    double accmeasNorm = norm(accmeas,R3);

    // #### Initial estimation ####
    // after starting the algorithm, the time constants are set to small
    // values to ensure fast convergence of the estimation
    // added 09/30/2015
    if ( (this->validAccDataCount+1)/this->sampleFreq/2  < this->tauAcc/2)  // start bias estimation after TauAcc/2 is reached to prevent
    {                                                                       // estimating the bias out of the error due to the initial orientation
        calculateCorrectionGains(                                           // set gains with min(count/2/freq, TauAcc) and min(count/2/freq, TauMag)
            fmin((this->validAccDataCount+1)/this->sampleFreq/2, this->tauAcc),
            fmin((this->validAccDataCount+1)/this->sampleFreq/2, this->tauMag),
            0                                                               // set Zeta to zero to prevent bias estimation
            );
    }
    else    // zeta can be user defined again
    {
        calculateCorrectionGains(                                          // set gains with min(count/2/freq, TauAcc) and min(count/2/freq, TauMag)
            fmin((this->validAccDataCount+1)/this->sampleFreq/2, this->tauAcc),
            this->tauMag < 100 ? fmin((this->validAccDataCount+1)/this->sampleFreq/2, this->tauMag) : this->tauMag, //FIXME: hack by Daniel for calibration demo
            this->zeta                                                      // set Zeta to the user defined value to enable bias estimation
            );
    }      // bias estimation on/off


    // #### Raw Value Rating ####
    // Do the raw value rating to reduce the correction gain if needed
    double  correctionRating = 1;                                           // default:no down rating of correction gain
    if (this->useMeasRawRating > 0)                                         // check if rating is enabled
    {
        memmove(this->ratingWindow, &this->ratingWindow[1],                 // move window/buffer one sample up
                (this->windowLength-1)*sizeof(double));
        this->ratingWindow[this->windowLength-1] = fabs(accmeasNorm-9.81);  // add the new sample to buffer

        double maxOfWind = 0;
        for(int iWind=0; iWind < this->windowLength; iWind++)               // calc max over window
        {                                                                   // "
            if (this->ratingWindow[iWind] > maxOfWind)                      // "
            {                                                               // "
                maxOfWind = this->ratingWindow[iWind];                      // "
            }                                                               // "
        }   // calc max over window END

        correctionRating = 1/(1+maxOfWind*useMeasRawRating);                // calculate correction rating (1: normal thrust, <1 down rating)
    }

	// #### Bias correction for gyro in imu frame ####
	for (int8_t iBias=0; iBias < R3; iBias++)
      {
		gyromeasBiasCorrected[iBias] = gyromeas[iBias] + this->storedBias[iBias];
      }

    // #### Gyro-based prediction #####
    // in: storedQuaternion  out: qGyro
	if ( 0 != gyromeasBiasCorrected[0] || 0 !=  gyromeasBiasCorrected[1]    // check gyro measurement to be valid
         || 0 != gyromeasBiasCorrected[2])                                  // "
      {
        double gyromeasNorm;                                                //
        double normedGyromeas[R3];                                          //
        // -----------------------------------------------------------------//
        gyromeasNorm        = norm(gyromeasBiasCorrected,R3);				// get norm to calculate prediction angle
		predictionAngle 	= gyromeasNorm / (2 * sampleFreq);				// calculate angle to correct in angle/axis
		normalize(gyromeasBiasCorrected, R3, gyromeasNorm, normedGyromeas);	// normalize for next step
		double dq_gyro[R4] 	= {	cos(predictionAngle), 						// specify quaternion for gyro correction
								sin(predictionAngle) * normedGyromeas[0],
								sin(predictionAngle) * normedGyromeas[1],
								sin(predictionAngle) * normedGyromeas[2]
							  };
		quaternionMultiply(this->storedQuaternion, dq_gyro, qGyro);         // correct stored quaternion by dq_gyro in IMU frame and save it to qGyro
	  }
    else // gyromeasBiasCorrected[0..2] == 0, gyro measurement was not valid
      {
        memmove(qGyro, this->storedQuaternion, R4*sizeof(double));          // just copy previous estimation because nothing has to be done
	  }

    // #### Accelerometer-based correction ####
    // in: qGyro   out: qGyroAcc
    memmove(qGyroAcc, qGyro, 4*sizeof(double));                             // copy estimation for case if no acc correction will be performed
	outputErrors[Incl] = 0;                                                 // set default error value of zero in case no correction wil be performed
    if (0 != accmeas[0]|| 0 != accmeas[1] || 0 != accmeas[2])               // check accmeas to be valid
      {
          this->validAccDataCount++;
        double normedAccmeas[R3];                                           // calculate norm of accmess
        // -----------------------------------------------------------------//
        quaternionCoordTransform(qGyro, this->gravrefFixedframe,            // transform gravitation reference into IMU frame
                                 this->gravrefImuframe);                    // "
        normalize(accmeas, R3, accmeasNorm, normedAccmeas);                 // normalize accmeas
        double preErrorAngleIncl = scalarProduct(normedAccmeas,             // calculate the error between measurement and reference
                                                 &this->gravrefImuframe[1]);// "
        if ( (1 > preErrorAngleIncl) && (-1 < preErrorAngleIncl) )          // perform correction only if accmeas and gravref_imuframe do NOT coincide
         {
            double errorAngleIncl = acos(preErrorAngleIncl);                // calculate error between reference and measurment
            double kp_acc         = this->correctionAccGain                 // calculate correction gain kp_acc
                                    * correctionRating;                     // "
            correctionAngle       = (kp_acc * errorAngleIncl) / 2;          // calculate angle for correction
            vectorCrossProduct(accmeas, &this->gravrefImuframe[1],          // rotation axis of correction is perpendicular to measurement and reference
                                correctionaxisImuframe);                    // "
            normalize (correctionaxisImuframe, R3,                          // normalize rotation axis
                        norm(correctionaxisImuframe,R3),				    // "
                        normedCorrectionaxisImuframe);                      // "

            double dqAcc[R4]  = {cos(correctionAngle),                      // specify quaternion for acc correction
                                 sin(correctionAngle) * normedCorrectionaxisImuframe[0], // build correction quaternion from axis and angle
                                 sin(correctionAngle) * normedCorrectionaxisImuframe[1],
                                 sin(correctionAngle) * normedCorrectionaxisImuframe[2]
                                };
            quaternionMultiply(qGyro, dqAcc, qGyroAcc);						// correct gyro corrected quaternion by acc correction
            // -------------------------------------------------------------//
            double ki_acc = kp_acc * kp_acc / (1-kp_acc)                    // calculate ki_acc for bias estimation
                            * this->correctionBiasGain;                     // "
            for(int8_t iBias=0; iBias < R3; iBias++)                        // update estimated gyro bias from accmeas
              {
                this->storedBias[iBias] += (ki_acc
                                            * errorAngleIncl
                                           * normedCorrectionaxisImuframe[iBias]);
              }
            outputErrors[Incl]     = errorAngleIncl;                        // set output to errorAngleIncl
         } // accmeas and gravref_imuframe did coincide
     }  // accmeas[0..2] != 0 , accelerometer data was valid

    // #### Magnetometer-based correction ####
    // in: qGyroAcc   out: qMagAccGyro
    memmove(qMagAccGyro, qGyroAcc, 4*sizeof(double));                       // sopy estimation for case if no mag correction will be performed
	outputErrors[Azi]  = 0;                                                 // set default error value of zero in case no correction wil be performed
    if (0 != magmeas[0] || 0 != magmeas[1] || 0 != magmeas[2])              // check accmeas to be valid
      {
        quaternionCoordTransform(qGyroAcc, this->gravrefFixedframe,         // recalculate IMU frame coordinates of vertical axis (might be skipped)
                                 this->gravrefImuframe);                    // "
        quaternionCoordTransform(qGyroAcc, this->magrefFixedFrame,          // transform magnetic field reference into IMU frame
                                 this->magrefImuframe);                     // "
        double magmeasNorm = norm(magmeas,R3);                              // calculate norm of magmeas
        double errorVecticalRef = scalarProduct(&this->gravrefImuframe[1],  // calculate the error between measurement and vertical reference
                                                magmeas);
        if (errorVecticalRef < magmeasNorm                                  // if measured magnetic field is vertical, perform NO correction
            && errorVecticalRef > -1*magmeasNorm)                           // "
         {
             double magmeasProjectedImuframe[R3];
             for (int8_t iPro = 0; iPro < R3; iPro++)                       // projection of magnetic measurement into horizontal plane
               {                                                            // "
                 magmeasProjectedImuframe[iPro] = magmeas[iPro]             // calculate projection per coordinate
                                                  - errorVecticalRef        // "
                                                  * this->gravrefImuframe[iPro+1];
               }                                                            // "
             double magmeasProjectedNormedImuframe[R3];
             normalize (magmeasProjectedImuframe, R3,                       // " normalize projection
                        norm(magmeasProjectedImuframe,R3),                  // "
       				    magmeasProjectedNormedImuframe);                    // "

            double preErrorAngleAzi =                                       // calculate the error between measurement and reference
                    scalarProduct(magmeasProjectedNormedImuframe,           // "
                                  &this->magrefImuframe[1]);                // "

             if (1 > preErrorAngleAzi && -1 < preErrorAngleAzi)             // if projected measurement and reference agree, perform NO correction
              {
                double errorAngleAzi = acos(preErrorAngleAzi);              // calculate error angle between reference and measurment
                double kp_mag    =  this->correctionMagGain;                // calculate correction gain  kp_mag
                correctionAngle  =  kp_mag * errorAngleAzi / 2;             // calculate angle for correction
                vectorCrossProduct(magmeasProjectedImuframe,				// rotation axis of correction is perpendicular to measurement and reference
                            	   &this->magrefImuframe[1],                // "
                                   correctionaxisImuframe);                 // "
                normalize (correctionaxisImuframe, R3,	                    // normalize rotation axis
                           norm(correctionaxisImuframe,R3),			        // "
                           normedCorrectionaxisImuframe);                   // "
                double dqMag[R4] = {cos(correctionAngle), 				    // build correction quaternion from axis and angle
                                    sin(correctionAngle) * normedCorrectionaxisImuframe[0],
                                    sin(correctionAngle) * normedCorrectionaxisImuframe[1],
                                    sin(correctionAngle) * normedCorrectionaxisImuframe[2]
                                   };
                quaternionMultiply(qGyroAcc, dqMag, qMagAccGyro);           // correct gyro&acc-corrected quaternion by mag correction
                // -------------------------------------------------------- //
                double ki_mag = kp_mag * kp_mag / (1-kp_mag)                // calculate ki_mag for bias estimation
                                * this->correctionBiasGain;
                for(int8_t iBias = 0; iBias <R3 ; iBias++)                  // update estimated gyro bias from magmeas
                  {
                    this->storedBias[iBias] += (ki_mag                      // estimate bias from magnetometer correction
                                                * errorAngleAzi
                                               * normedCorrectionaxisImuframe[iBias]);
                  }
                outputErrors[Azi] = errorAngleAzi;                          // set output to errorAngleAzi
              } // projected measurement and reference didn't agreed
         } // measured magnetic field was not is vertical
     } // magmeas was valid


	normalize(qMagAccGyro, R4, norm(qMagAccGyro,R4),storedQuaternion);		// normalize estimated quaternion before output to account for numerical drifts

    if (storedQuaternion[0] < 0)                                            // make sure the first entry of the quaternion is always positive (for comfort)
    {                                                                       // reason: there is an ambiguity in the quaternion: -q = q for orientation
        for (int8_t iInv = 0; iInv < R4; iInv++)                            // "
            storedQuaternion[iInv] = -1*storedQuaternion[iInv];             // storedQuaternion = -storedQuaternion to ensure storedQuaternion[0] >= 0
    }                                                                       // "

	memmove(outputQuaternion, this->storedQuaternion, sizeof(double)*R4);   // set estimated quaternion to output: correctedQuaternion
	memmove(outputEstimatedBias, this->storedBias, sizeof(double)*R3);      // set estimated gyro bias to output: correctedGyroBias
    return SUCCESS;
} // CCsgOriEstIMU::updateImprovedAHRSIMU

void CCsgOriEstIMU::setRating(double rating)  // ADDED BY DL
{
    this->useMeasRawRating = rating;
}
