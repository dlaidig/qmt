#ifndef MadgwickB_h
#define MadgwickB_h

// This implementation is based on Appendix A and Appendix B of
// S. Madgwick. An efficient orientation filter for inertial and inertial/magnetic sensor arrays.
// Report x-io and University of  Bristol (UK), 25, 2010. (madgwick_internal_report.pdf).
//
// The code has been transformed into a class and both update functions have been merged by Daniel Laidig <laidig@control.tu-berlin.de>.
// Note that bias estimation is only included in the AHRS version of the original code (on purpose, as stated in the internal report)
// and when using this class without magnetometer data, zeta should be set to 0.0.
// This implementation will still allow for bias estimation to be performed without magnetometer updates for testing purposes.
//
// The magnetometer-free update is equivalent to the implementation found in MadgwickAHRS.cpp. The update with magnetometer
// is different from this version is different as b_x and b_z are estimated at the end of the update step. Note that the original
// implementation did not normalize the estimated b_x and b_z and did therefore not work properly when the magnetometer measurements
// are given in µT. This has been changed in this implementation.

class MadgwickB {
public:
    MadgwickB(float deltat, float beta, float zeta);

    void update(float w_x, float w_y, float w_z, float a_x, float a_y, float a_z, float m_x, float m_y, float m_z, bool use_mag);

    float deltat; // sampling time in seconds
    float beta; // gain, default value in original source: sqrt(3/4)*deg2rad(5), with gyroMeasError = 5°/s
    float zeta; // bias estimation gain, default value in original source: sqrt(3/4)*deg2rad(0.2), with gyroMeasDrift = 0.2°/s/s

    float SEq_1, SEq_2, SEq_3, SEq_4; // estimated orientation quaternion elements with initial conditions
    float b_x, b_z;                   // reference direction of flux in earth frame
    float w_bx, w_by, w_bz;           // estimate gyroscope biases error
};

#endif
