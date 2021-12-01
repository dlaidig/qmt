#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
"""
Interactive offline full body motion tracking demo.

The data processing is based on the code found in full_body_tracking_basic_example.py and
full_body_tracking_advanced_example.py. A custom webapp is used that allows for the side-by-side comparison of different
versions and for interactively changing the algorithm parameters.
"""
import copy
from base64 import b64decode
import numpy as np
import qmt

from full_body_tracking_basic_example import tunedParams
from full_body_tracking_advanced_example import mergeSettings, estimateOrientations, resetAlignment, \
    estimateJointAxes, resetHeading, headingCorrection, markers


def applyTunedParams(data, settings):
    # apply manually tuned segment-to-sensor orientation and heading offset to each segment
    for segment in settings:
        qSeg2Imu = qmt.normalized(tunedParams.get(f'{segment}.q_segment2sensor', [1, 0, 0, 0]))
        deltaQuat = qmt.deltaQuat(np.deg2rad(tunedParams.get(f'{segment}.heading_offset', 0)))
        data[f'{segment}.quat_seg_tuned'] = qmt.qmult(deltaQuat, qmt.qmult(data[f'{segment}.quat_imu'], qSeg2Imu))


def processData(data, settings):
    estimateOrientations(data, settings)
    applyTunedParams(data, settings)
    resetAlignment(data, settings)
    estimateJointAxes(data, settings)
    resetHeading(data, settings)
    headingCorrection(data, settings, saveDebug=True)


async def onParams(webapp, params):
    print('new params: ', params)
    print('changes:')
    paramStruct = qmt.Struct(copy.deepcopy(params))
    paramStruct.createArrays()
    defaultStruct = qmt.Struct(mergeSettings())
    defaultStruct.createArrays()
    paramStruct.diff(defaultStruct)

    data = webapp.data
    if webapp.show in ('window', 'widget'):
        # run processing in thread to avoid blocking the Qt event loop and freezing the webapp window
        import asyncio
        import qasync
        loop = asyncio.get_running_loop()
        with qasync.QThreadExecutor(1) as executor:
            await loop.run_in_executor(executor, processData, data, params)
        # processData(data, params)
    else:
        processData(data, params)
    data.save('example_output/full_body_demo_tuned.mat', makedirs=True)
    webapp.data = data
    print('data updated!')


def onCommand(_, command):
    if command[0] == 'frame':
        # example command to create video from saved frames (adjust start number):
        # ffmpeg -framerate 100 -start_number 726 -i frame_%06d.png  -c:v libx264 -r 30 video.mp4
        ind = command[1]
        data_uri = command[2]
        header, encoded = data_uri.split(',', 1)
        png = b64decode(encoded)
        with open(f'example_output/frame_{ind:06d}.png', 'wb') as f:
            f.write(png)
    else:
        print('unknown command received:', command[0])


def main():
    data = qmt.Struct.load('full_body_example_data.mat')
    settings = mergeSettings()

    processData(data, settings)

    for key in data.leafKeys():  # create copy of data that can be used for comparision when parameters are changed
        if '.quat_seg_' in key:
            data[key+'_default'] = data[key]

    config = {
        'base': 'human',
        'segments': {
            segment: {
                'signal': f'{segment}.quat_seg_deltaCorr',
                'q_segment2sensor': data[f'{segment}.q_segment2sensor'],
                'imubox_cs': 'FLU',  # the x-axis of the IMU is pointing forward, the y-axis to the left
            } for segment in settings
        },
        'debug_mode': True,
        'markers': markers,
        'signals': {
            'Manual tuning': {name: f'{name}.quat_seg_tuned' for name in settings},
            'Manual tuning (default params)': {name: f'{name}.quat_seg_tuned_default' for name in settings},
            'Heading reset': {name: f'{name}.quat_seg_resetHeading' for name in settings},
            'Heading reset (default params)': {name: f'{name}.quat_seg_resetHeading_default' for name in settings},
            'Heading correction': {name: f'{name}.quat_seg_deltaCorr' for name in settings},
            'Heading correction (default params)': {name: f'{name}.quat_seg_deltaCorr_default' for name in settings},
        },
        'defaultSettings': settings,
    }
    data.save('example_output/full_body_demo.mat', makedirs=True)
    qmt.Struct(config).save('example_output/full_body_demo_config.json')
    webapp = qmt.Webapp('/demo/full-body-tracking', data=data, config=config, quiet=True)
    webapp.on('params', onParams)
    webapp.on('command', onCommand)
    webapp.run()


if __name__ == '__main__':
    main()
