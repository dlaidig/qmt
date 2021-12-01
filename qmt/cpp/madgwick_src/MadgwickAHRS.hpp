// This implementation is based on the code found in original_downloads/madgwick_algorithm_c.zip.
// The code has been transformed into a class by Daniel Laidig <laidig@control.tu-berlin.de>.
// Furthermore, the fast inverse square-root implementation was disabled.

//=====================================================================================================
// MadgwickAHRS.h
//=====================================================================================================
//
// Implementation of Madgwick's IMU and AHRS algorithms.
// See: http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
//
// Date			Author          Notes
// 29/09/2011	SOH Madgwick    Initial release
// 02/10/2011	SOH Madgwick	Optimised for reduced CPU load
//
//=====================================================================================================
#ifndef MadgwickAHRS_h
#define MadgwickAHRS_h

//----------------------------------------------------------------------------------------------------
// Variable declaration

//extern volatile float beta;				// algorithm gain
//extern volatile float q0, q1, q2, q3;	// quaternion of sensor frame relative to auxiliary frame

//---------------------------------------------------------------------------------------------------
// Function declarations

class MadgwickAHRS {
public:
    MadgwickAHRS(float beta, float sampleFreq);

    void update(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz);
    void updateIMU(float gx, float gy, float gz, float ax, float ay, float az);

    float beta;
    float q0;
    float q1;
    float q2;
    float q3;
    float sampleFreq;
};

#endif
//=====================================================================================================
// End of file
//=====================================================================================================
