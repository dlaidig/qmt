#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
"""
Offline full body motion tracking example.

This example shows how the qmt framework can be used to perform offline full body 6D motion tracking and animate a 3D
avatar with just a few lines of code. This script is kept as short and simple as possible. For a more complex example,
see the file full_body_tracking_advanced_example.py.
"""
import qmt
import numpy as np

# manually tuned sensor-to-segment orientations and heading offsets
# (use the "copy config changes" button in the boxmodel webapp to obtain those values)
tunedParams = qmt.Struct({
    'hip':  {'heading_offset': 0.0, 'q_segment2sensor': [0.542, 0.542, 0.455, -0.455]},
    'lower_back': {'heading_offset': -30.0, 'q_segment2sensor': [0.455, 0.455, 0.542, -0.542]},
    'upper_back': {'heading_offset': 15.0, 'q_segment2sensor': [0.579, 0.579, 0.406, -0.406]},
    'head': {'heading_offset': -45.0, 'q_segment2sensor': [0.406, 0.406, 0.579, -0.579]},
    'upper_arm_left': {'heading_offset': -89.5, 'q_segment2sensor': [0.488, 0.354, 0.449, -0.66]},
    'forearm_left': {'heading_offset': -95.0, 'q_segment2sensor': [0.482, 0.517, 0.517, -0.482]},
    'hand_left': {'heading_offset': -85.0, 'q_segment2sensor': [0.5, 0.5, 0.5, -0.5]},
    'upper_arm_right': {'heading_offset': 60.0, 'q_segment2sensor': [0.43, 0.467, 0.617, -0.465]},
    'forearm_right': {'heading_offset': 80.0, 'q_segment2sensor': [0.542, 0.455, 0.455, -0.542]},
    'hand_right': {'heading_offset': 54.0, 'q_segment2sensor': [0.46, 0.478, 0.622, -0.416]},
    'upper_leg_left': {'heading_offset': -140.0, 'q_segment2sensor': [0.172, -0.73, -0.661, 0.014]},
    'lower_leg_left': {'heading_offset': 105.0, 'q_segment2sensor': [0.641, -0.299, -0.242, -0.664]},
    'foot_left': {'heading_offset': -150.0, 'q_segment2sensor': [-0.205, 0.158, 0.766, 0.588]},
    'upper_leg_right': {'heading_offset': 80.0, 'q_segment2sensor': [0.703, -0.182, 0.048, -0.686]},
    'lower_leg_right': {'heading_offset': -155.0, 'q_segment2sensor': [0.282, -0.681, -0.624, -0.259]},
    'foot_right': {'heading_offset': -150.0, 'q_segment2sensor': [-0.158, 0.205, 0.588, 0.766]},
})
tunedParams.createArrays()  # convert all q_segment2sensor quaternions from lists to numpy arrays


if __name__ == '__main__':
    # load the raw IMU data
    data = qmt.Struct.load('full_body_example_data.mat')

    # the data only contains a time vector. calculate the sampling time from it.
    timediff = np.diff(data['t'])
    Ts = timediff[0]
    assert np.allclose(timediff, Ts)  # make sure the sampling time is constant to avoid surprises

    # list of segment names that IMUs are attached to
    segments = [k for k in data.keys() if k != 't']

    # run orientation estimation for each IMU
    params = {'Ts': Ts, 'tauAcc': 1, 'zeta': 0, 'accRating': 1}
    for name in segments:
        quat = qmt.oriEstIMU(gyr=data[f'{name}.gyr'], acc=data[f'{name}.acc'], params=params)
        data[f'{name}.quat_imu'] = quat

    # apply those quaternions to each segment
    for name in segments:
        qSeg2Imu = qmt.normalized(tunedParams.get(f'{name}.q_segment2sensor', [1, 0, 0, 0]))
        data[f'{name}.quat_seg'] = qmt.qmult(data[f'{name}.quat_imu'], qSeg2Imu)

    # correct heading for each segment
    for name in segments:
        deltaQuat = qmt.deltaQuat(np.deg2rad(tunedParams.get(f'{name}.heading_offset', 0)))
        data[f'{name}.quat_seg_adjusted'] = qmt.qmult(deltaQuat, data[f'{name}.quat_seg'])

    # create config for visualization
    data['config'] = {
        'base': 'human',
        'segments': {
            name: {
                'signal': f'{name}.quat_seg_adjusted',
                'heading_offset': tunedParams.get(f'{name}.heading_offset', 0),
                'q_segment2sensor': tunedParams.get(f'{name}.q_segment2sensor', [1, 0, 0, 0]),
                'imubox_cs': 'FLU',  # the x-axis of the IMU is pointing forward, the y-axis to the left
            } for name in segments
        },
        'debug_mode': True,
    }

    # save the processed data (including the config) to a mat file
    data.save('example_output/full_body_results.mat', makedirs=True)

    # visualize the motion with a box model
    webapp = qmt.Webapp('/view/boxmodel', data=data, quiet=True)
    webapp.run()
