#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
"""
Offline full body motion tracking example.

This example performs offline full body 6D motion tracking using more advanced algorithms. For a simpler example,
see the file full_body_tracking_basic_example.py.
"""
import qmt
import numpy as np


# all body segments with the respective parents
parents = {
    'hip': None,
    'lower_back': 'hip',
    'upper_back': 'lower_back',
    'head': 'upper_back',
    'upper_arm_left': 'upper_back',
    'forearm_left': 'upper_arm_left',
    'hand_left': 'forearm_left',
    'upper_arm_right': 'upper_back',
    'forearm_right': 'upper_arm_right',
    'hand_right': 'forearm_right',
    'upper_leg_left': 'hip',
    'lower_leg_left': 'upper_leg_left',
    'foot_left': 'lower_leg_left',
    'upper_leg_right': 'hip',
    'lower_leg_right': 'upper_leg_right',
    'foot_right': 'lower_leg_right',
}

# time markers to show in the playback timeline
markers = [
    dict(pos=40, end=44, name='standing', col='C2'),
    dict(pos=52, end=79, name='sitting', col='C2'),
    dict(pos=95, end=115, name='walking', col='C0'),
    dict(pos=115, name='turn', col='C3'),
    dict(pos=116, end=138.5, name='walking (2)', col='C0'),
    dict(pos=138.5, name='turn (2)', col='C3'),
]

# settings for qmt.resetAlignment
resetAlignmentSettings = {
    'hip': dict(x=[0, 0, -1], xCs=0, y=[0, 0, 1], yCs=-1, exactAxis='y'),
    'lower_back': dict(x=[0, 0, -1], xCs=0, y=[0, 0, 1], yCs=-1, exactAxis='y'),
    'upper_back': dict(x=[0, 0, -1], xCs=0, y=[0, 0, 1], yCs=-1, exactAxis='y'),
    'head': dict(x=[0, 0, -1], xCs=0, y=[0, 0, 1], yCs=-1, exactAxis='y'),
    'upper_arm_left': dict(x=[0, -1, -1], xCs=0, y=[0, 0, 1], yCs=-1, exactAxis='y'),
    'upper_arm_right': dict(x=[0, 1, -1], xCs=0, y=[0, 0, 1], yCs=-1, exactAxis='y'),
    'forearm_left': dict(x=[0, 0, -1], xCs=0, y=[0, 0, 1], yCs=-1),
    'forearm_right': dict(x=[0, 0, -1], xCs=0, y=[0, 0, 1], yCs=-1),
    'hand_left': dict(x=[0, 0, -1], xCs=0, y=[0, 0, 1], yCs=-1),
    'hand_right': dict(x=[0, 0, -1], xCs=0, y=[0, 0, 1], yCs=-1),
}

# settings for qmt.jointAxisEstHingeOlsson, for the joint between the given segment and the respective parent
jointEstSettings = {
    'foot_left': dict(wa=100, wg=0.1, useSampleSelection=True, angRateEnergyThreshold=51, winSize=41, dataSize=7500,
                      tol=1e-8, flip=False, quiet=True),
    'lower_leg_left': dict(wa=10, wg=3, useSampleSelection=True, angRateEnergyThreshold=51, winSize=41, dataSize=7500,
                           tol=1e-8, flip=True, applyToParent=True, flipParent=True, quiet=True),
    'foot_right': dict(wa=100, wg=0.1, useSampleSelection=True, angRateEnergyThreshold=51, winSize=41, dataSize=7500,
                       tol=1e-8, flip=False, quiet=True),
    'lower_leg_right': dict(wa=10, wg=3, useSampleSelection=True, angRateEnergyThreshold=51, winSize=41, dataSize=7500,
                            tol=1e-8, flip=True, applyToParent=True, flipParent=True, quiet=True)
}

# settings for qmt.resetHeading, for the joint between the given segment and the respective parent
resetHeadingSettings = {
    'upper_arm_left': dict(deltaOffset=np.deg2rad(-90)),
    'upper_arm_right': dict(deltaOffset=np.deg2rad(90)),
}

# settings for qmt.headingCorrection, for the joint between the given segment and the respective parent
deltaCorrectionSettings = {
    'hip': dict(),
    'lower_back': dict(joint=[0, 0, 1]),
    'upper_back': dict(joint=[0, 0, 1]),
    'upper_leg_left': dict(joint=[0, 0, 1]),
    'lower_leg_left': dict(joint=[0, 0, 1]),
    'foot_left': dict(joint=[0, 0, 1], est_settings={'startRating': 0.5, 'stillnessRating': 0.5}),
    'upper_leg_right': dict(joint=[0, 0, 1]),
    'lower_leg_right': dict(joint=[0, 0, 1]),
    'foot_right': dict(joint=[0, 0, 1]),
    'head': dict(
        joint=[[1, 0, 0], [0, 0, 1]],
        est_settings=dict(angleRanges=np.deg2rad(np.array([[-80, 80], [-70, 70], [-30, 80]], float)),
                          convention='yxz', useRomConstraints=True)
    ),
    'upper_arm_left': dict(
        joint=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        joint_info=dict(angle_ranges=np.deg2rad(np.array([[-5, 40], [-90, -70], [-10, 40]], float)), convention='xyz')
    ),
    'forearm_left': dict(
        joint=[[1, 0, 0], [0, 1, 0]],
        est_settings={'ratingMin': 0.1, 'startRating': 0, 'stillnessRating': 0}
    ),
    'hand_left': dict(
        joint=[[0, 0, 1], [0, 1, 0]],
        est_settings={'ratingMin': 0.1, 'startRating': 0.3, 'stillnessRating': 0.3}
    ),
    'hand_right': dict(
        joint=[[0, 0, 1], [0, 1, 0]],
        est_settings={'ratingMin': 0.1, 'startRating': 0.3, 'stillnessRating': 0.3}
    ),
    'forearm_right': dict(
        joint=[[1, 0, 0], [0, 1, 0]],
        est_settings={'ratingMin': 0.1, 'startRating': 0, 'stillnessRating': 0}
    ),
    'upper_arm_right': dict(
        joint=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        joint_info=dict(angle_ranges=np.deg2rad(np.array([[-30, 0], [80, 105], [-10, 40]], float)), convention='xyz')
    ),
}


def mergeSettings():
    return {
        name: {
            'resetAlignmentSettings': resetAlignmentSettings.get(name, {}),
            'jointEstSettings': jointEstSettings.get(name, {}),
            'resetHeadingSettings': resetHeadingSettings.get(name, {}),
            'deltaCorrectionSettings': deltaCorrectionSettings.get(name, {}),
        } for name, parent in parents.items()
    }


def estimateOrientations(data, settings, plot=False):
    timediff = np.diff(data['t'])  # the data only contains a time vector. calculate the sampling rate from it
    assert np.allclose(timediff, timediff[0])  # make sure the sampling time is constant to avoid surprises
    # run orientation estimation for each IMU
    defaults = {'Ts': timediff[0], 'tauAcc': 1, 'zeta': 0, 'accRating': 1}
    for name in settings:
        qmt.setupDebugPlots(title=name)  # show the segment name in the plot title
        params = qmt.setDefaults(settings[name].get('oriEst', {}), defaults)
        quat = qmt.oriEstIMU(gyr=data[f'{name}.gyr'], acc=data[f'{name}.acc'], params=params, plot=plot)
        quat[0:100] = quat[100]  # remove initial convergence phase
        data[f'{name}.quat_imu'] = quat


def resetAlignment(data, settings, plot=False):
    reset = np.zeros(data['t'].shape)
    reset[0] = 1
    for name in settings:
        params = settings[name].get('resetAlignmentSettings', {})
        if np.allclose(params.get('x', [0, 0, 0]), 0) and np.allclose(params.get('y', [0, 0, 0]), 0) and \
                np.allclose(params.get('z', [0, 0, 0]), 0):
            continue

        qmt.setupDebugPlots(title=name)
        q_out = qmt.resetAlignment(data[f'{name}.quat_imu'][None], reset, **params, plot=plot)
        data[f'{name}.quat_seg'] = q_out[0]
        data[f'{name}.q_segment2sensor'] = qmt.qrel(data[f'{name}.quat_imu'][0], data[f'{name}.quat_seg'][0])


def estimateJointAxes(data, settings, plot=False):
    defaults = {'tol': 1e-8, 'quiet': True}
    for name, parent in parents.items():
        estSettings = qmt.setDefaults(settings[name].get('jointEstSettings', {}), defaults, check=False)
        if 'wa' not in estSettings:
            continue
        qmt.setupDebugPlots(title=f'{parent}/{name}')
        jhat1, jhat2 = qmt.jointAxisEstHingeOlsson(data[f'{parent}.acc'], data[f'{name}.acc'], data[f'{parent}.gyr'],
                                                   data[f'{name}.gyr'], estSettings=estSettings, plot=plot)
        flip = estSettings.get('flip')
        flipParent = estSettings.get('flipParent')
        for segment, jhat, flip in ((parent, jhat1, flipParent), (name, jhat2, flip)):
            if segment == parent and not estSettings.get('applyToParent'):
                continue
            jhat = -jhat.squeeze() if flip else jhat.squeeze()
            qBS = qmt.quatFrom2Axes(z=jhat, y=qmt.normalized(data[f'{segment}.acc'][100, :]), exactAxis='y')
            data[f'{segment}.quat_seg'] = qmt.qmult(data[f'{segment}.quat_imu'], qBS)
            data[f'{segment}.q_segment2sensor'] = qBS


def resetHeading(data, settings, plot=False):
    reset = np.zeros(data['t'].shape)
    reset[0] = 1

    for name, parent in parents.items():
        if parent is None:
            data[f'{name}.quat_seg_resetHeading'] = data[f'{name}.quat_seg']
            continue
        q = [data[f'{parent}.quat_seg_resetHeading'], data[f'{name}.quat_seg']]
        deltaOffset = settings[name].get('resetHeadingSettings', {}).get('deltaOffset', 0)
        qmt.setupDebugPlots(title=f'{parent}/{name}')
        out = qmt.resetHeading(q, reset, base=0, deltaOffset=deltaOffset, plot=plot)
        data[f'{name}.quat_seg_resetHeading'] = out[1]


def headingCorrection(data, settings, plot=False, saveDebug=True):
    delta = {}
    t = data['t']
    for name, parent in parents.items():
        if parent is None:
            delta[name] = np.zeros(t.shape[0])
            continue
        gyr1 = data[f'{parent}.gyr']
        gyr2 = data[f'{name}.gyr']
        quat1 = data[f'{parent}.quat_seg']
        quat2 = data[f'{name}.quat_seg']

        params = settings[name]['deltaCorrectionSettings']

        joint = params['joint']
        joint_info = params.get('joint_info', {})
        est_settings = params.get('est_settings', {})

        print(f'headingCorrection: {parent}/{name}')
        qmt.setupDebugPlots(title=f'{parent}/{name}')
        out = qmt.headingCorrection(gyr1, gyr2, quat1, quat2, t, joint, joint_info, est_settings, plot=plot)
        delta_filt = out[2]
        if saveDebug:
            data[f'{name}.delta'] = out[1]
            data[f'{name}.delta_filt'] = out[2]
            data[f'{name}.rating'] = out[3]
        delta[name] = delta[parent] + delta_filt

    for name in parents:
        deltaQuat = qmt.deltaQuat(delta[name])
        data[f'{name}.quat_seg_deltaCorr'] = qmt.qmult(deltaQuat, data[f'{name}.quat_seg'])


def main():
    # disable/enable the creation of debug plots
    plot = False
    # plot = True

    # disable/enable continuous constraint-based heading correction (if False, headingReset is used)
    useHeadingCorrection = False
    # useHeadingCorrection = True

    if plot:
        # ensure that the webapp viewer is initialized before any plots are created
        qmt.Webapp.initialize()
        # setup debug plots: save to one multipage pdf and increase the size
        qmt.setupDebugPlots(mode='pdfpages', filename='example_output/full_body_debug_plots.pdf', figsize_cm=(29.7, 21))

    data = qmt.Struct.load('full_body_example_data.mat')
    settings = mergeSettings()

    estimateOrientations(data, settings, plot)
    resetAlignment(data, settings, plot)
    estimateJointAxes(data, settings, plot)
    if not useHeadingCorrection:
        resetHeading(data, settings, plot)
    else:
        headingCorrection(data, settings, plot)

    qmt.setupDebugPlots(mode='show')  # close the PDF file before opening the visualization
    quatSignal = 'quat_seg_deltaCorr' if useHeadingCorrection else 'quat_seg_resetHeading'
    data['config'] = {
        'base': 'human',
        'segments': {
            name: {
                'signal': f'{name}.{quatSignal}',
                'q_segment2sensor': data[f'{name}.q_segment2sensor'],
                'imubox_cs': 'FLU',  # the x-axis of the IMU is pointing forward, the y-axis to the left
            } for name in settings
        },
        'debug_mode': True,
        'markers': markers,
    }
    data.save('example_output/full_body_results_advanced.mat', makedirs=True)
    webapp = qmt.Webapp('/view/boxmodel', data=data, quiet=True)

    webapp.run()


if __name__ == '__main__':
    main()
