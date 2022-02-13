#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import math
import os
import logging
import argparse
import sys

import numpy as np

import qmt
from vqf import BasicVQF

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SimpleMagCalibBlock(qmt.Block):
    def __init__(self, Ts, duration):
        super().__init__()
        self.params['apply'] = True
        self.Ts = Ts
        self.duration = duration
        self.collecting = False
        self.buffer = None
        self.startTime = None
        self.gain = np.ones((3,))
        self.bias = np.zeros((3,))
        self.calibrated = False
        self.info = None

        self.filename = None

    def command(self, command):
        assert command == ['start'], command
        self.start()

    def step(self, gyr, acc, mag, sensorid):
        if not self.filename:
            self.filename = 'calib/mag_{}.json'.format(sensorid)
            if os.path.exists(self.filename):
                self.loadFromFile()

        state = float(self.calibrated)

        if self.collecting:
            self.buffer['gyr'].append(gyr)
            self.buffer['acc'].append(acc)
            self.buffer['mag'].append(mag)

            N = len(self.buffer['gyr'])
            duration = N*self.Ts
            progress = duration / self.duration
            state = progress

            if progress >= 1:
                self.stop()

        if self.params['apply']:
            mag = self.gain*mag - self.bias

        return mag, dict(state=state, gain=self.gain, bias=self.bias)

    def start(self):
        self.collecting = True
        self.buffer = dict(gyr=[], acc=[], mag=[])

    def stop(self):
        self.collecting = False
        N = len(self.buffer['acc'])
        duration = N*self.Ts

        acc = np.array(self.buffer['acc'], float)
        gyr = np.array(self.buffer['gyr'], float)
        mag = np.array(self.buffer['mag'], float)
        gain, bias, success, debug = qmt.calibrateMagnetometerSimple(gyr, acc, mag, self.Ts, debug=True)

        self.gain = gain
        self.bias = bias
        self.calibrated = success
        self.info = dict(
            duration=duration,
            ranges=debug['ranges'],
            before=debug['before'],
            init=debug['init'],
            result=debug['result'],
        )
        self.saveToFile()

    def loadFromFile(self):
        data = qmt.Struct.load(self.filename)
        self.gain = np.array(data['gain'])
        self.bias = np.array(data['bias'])
        self.calibrated = data['success']

    def saveToFile(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        data = {'gain': self.gain, 'bias': self.bias, 'success': self.calibrated, 'info': self.info}
        qmt.Struct(data).save(self.filename, indent=True)


class MagFieldAnalysisBlock(qmt.Block):
    def __init__(self, Ts):
        super().__init__()
        self.lastHeading = 0

        self.vqf = BasicVQF(Ts)
        self.lpBlock = qmt.LowpassFilterBlock(Ts, tau=10)

    def step(self, gyr, acc, mag):
        self.vqf.update(gyr, acc)
        quat = self.vqf.getQuat6D()

        mag_global = BasicVQF.quatRotate(quat, mag)
        BasicVQF.normalize(mag_global)

        dip = 90-np.rad2deg(math.acos(mag_global[2]))
        if np.isnan(dip):
            dip = 0

        heading = math.atan2(mag_global[1], mag_global[0])
        heading = self.lastHeading + qmt.wrapToPi(heading - self.lastHeading)  # unwrap
        heading_lp = self.lpBlock.step(heading)
        self.lastHeading = heading_lp

        return dict(
            norm=BasicVQF.norm(mag),
            dip=dip,
            heading_hp=np.rad2deg(heading - heading_lp),
        )


class BiasSimulationBlock(qmt.Block):
    def __init__(self):
        super().__init__()
        self.params['bias'] = np.zeros((3,), float)
        self.params['enabled'] = True

    def setParams(self, params):
        for name, value in params.items():
            if name == 'bias':
                self.params['bias'] = np.deg2rad(np.asarray(params['bias'], float))
                assert self.params['bias'].shape == (3,), self.params['bias'].shape
            elif name == 'enabled':
                self.params['enabled'] = value
            else:
                raise RuntimeError(f'invalid parameter name: {name}')

    def step(self, gyr):
        if not self.params['enabled']:
            return gyr
        return gyr + self.params['bias']


class Demo(qmt.Block):
    def __init__(self, Ts):
        super().__init__()
        self.params['algorithmA'] = 'Madgwick'
        self.params['algorithmB'] = 'Madgwick'
        self.params['applyMagCalibA'] = True
        self.params['applyMagCalibB'] = True
        self.Ts = Ts
        self.i = -1
        self.wsDownsampleFactor = max(1, int(0.04/Ts))  # downsample websocket communication to 25 Hz
        logger.info(f'sensor sampling rate {1/self.Ts:.1f} Hz, downsampling by factor {self.wsDownsampleFactor} to '
                    f'{1/self.Ts/self.wsDownsampleFactor:.1f} Hz')

        self.magCalib = SimpleMagCalibBlock(Ts, duration=10)
        self.magFieldAnalysis = MagFieldAnalysisBlock(Ts)
        self.magFieldAnalysisRaw = MagFieldAnalysisBlock(Ts)
        self.biasSimA = BiasSimulationBlock()
        self.biasSimB = BiasSimulationBlock()

        self.algBlocks = {
            'A': self.createOriEstBlocks(),
            'B': self.createOriEstBlocks(),
        }
        self.alg = {
            'A': self.algBlocks['A'][self.params['algorithmA']],
            'B': self.algBlocks['B'][self.params['algorithmB']],
        }

        self.lpGyrA = qmt.LowpassFilterBlock(Ts, tau=0.5)
        self.lpGyrB = qmt.LowpassFilterBlock(Ts, tau=0.5)
        self.lpDisAngleA = qmt.LowpassFilterBlock(Ts, tau=0.5)
        self.lpDisAngleB = qmt.LowpassFilterBlock(Ts, tau=0.5)

    def createOriEstBlocks(self):
        return {
            'Madgwick': qmt.OriEstMadgwickBlock(self.Ts),
            'Mahony': qmt.OriEstMahonyBlock(self.Ts),
            'OriEstIMU': qmt.OriEstIMUBlock(self.Ts),
            'VQF': qmt.OriEstVQFBlock(self.Ts),
        }

    def setParams(self, params):
        for name, value in params.items():
            if self.params.get(name, None) == value:
                continue
            logger.info(f'setting {name} to {value}')
            if name.startswith('algorithm'):
                imu = name[len('algorithm'):]
                self.alg[imu] = self.algBlocks[imu][value]
                self.alg[imu].command(['reset'])
            elif name in ('applyMagCalibA', 'aplyMagCalibB'):
                pass  # store in this class
            elif name == 'biasSim':
                self.biasSimA.setParams({'bias': value})
                self.biasSimB.setParams({'bias': value})
            elif name == 'biasSimEnabledA':
                self.biasSimA.setParams({'enabled': value})
            elif name == 'biasSimEnabledB':
                self.biasSimB.setParams({'enabled': value})
            elif name.startswith('params_'):
                _, imu, alg = name.split('_', maxsplit=2)
                self.algBlocks[imu][alg].setParams(value)
            elif name.startswith('param_'):
                _, imu, alg, name = name.split('_', maxsplit=3)
                self.algBlocks[imu][alg].setParams({name: value})
            else:
                logger.warning(f'ignored parameter {name}={value}')
            self.params[name] = value

    def command(self, command):
        if command == ['startMagCalib']:
            self.magCalib.command(['start'])
        elif command == ['resetA']:
            self.alg['A'].command(['reset'])
        elif command == ['resetB']:
            self.alg['B'].command(['reset'])
        else:
            logger.warning(f'ignoring command {command}')

    def step(self, sample):
        sample['mag_raw'] = sample['mag1']
        sample['mag1'], sample['mag_calib'] = self.magCalib.step(sample['gyr1'], sample['acc1'], sample['mag1'],
                                                                 sample['sensorid1'])
        sample['mag_field'] = self.magFieldAnalysis.step(sample['gyr1'], sample['acc1'], sample['mag1'])
        sample['mag_field_raw'] = self.magFieldAnalysisRaw.step(sample['gyr1'], sample['acc1'], sample['mag_raw'])
        sample['gyrA'] = self.biasSimA.step(sample['gyr1'])
        sample['gyrB'] = self.biasSimB.step(sample['gyr1'])

        sample['magA'] = sample['mag1'] if self.params['applyMagCalibA'] else sample['mag_raw']
        sample['magB'] = sample['mag1'] if self.params['applyMagCalibB'] else sample['mag_raw']
        sample['quatA'], sample['debugA'] = self.alg['A'].step(sample['gyrA'], sample['acc1'], sample['magA'],
                                                               debug=True)
        sample['quatB'], sample['debugB'] = self.alg['B'].step(sample['gyrB'], sample['acc1'], sample['magB'],
                                                               debug=True)

        sample['gyrA_lp'] = self.lpGyrA.step(sample['gyrA'])
        sample['gyrB_lp'] = self.lpGyrB.step(sample['gyrB'])
        sample['disAngleA_lp'] = self.lpDisAngleA.step(sample['debugA']['disAngle'])
        sample['disAngleB_lp'] = self.lpDisAngleB.step(sample['debugB']['disAngle'])

        sample['algorithmA'] = self.params['algorithmA']
        sample['algorithmB'] = self.params['algorithmB']

        if sample['t'] % 10 < 0.0001:
            logger.info(f'battery: {sample.get("battery1", -1)} %')
        self.i += 1
        if self.i % self.wsDownsampleFactor == 0:
            return sample
        return []


def main():
    parser = argparse.ArgumentParser(description='Orientation estimation demo.')
    parser.add_argument('--headless', action='store_true', help='do not open window')
    parser.add_argument('--bind', action='store_true', help='bind on 0.0.0.0')
    parser.add_argument('--port', default=8000, type=int, help='web server port, default: %(default)s')
    parser.add_argument('--dev', action='store_true', help='enable dev tools and use development server on port 3000')
    parser.add_argument('--cs', default='RFU', help='IMU coordinate system, default: %(default)s')
    parser.add_argument('source', nargs='?', default='{"class": "qmt.DummyImuDataSource", "Ts": 0.01}',
                        help='data source configuration as JSON string, default: %(default)s')
    args = parser.parse_args()

    config = {'cs': args.cs}

    webapp = qmt.Webapp('/demo/orientation-estimation', config=config)
    if args.headless:
        webapp.show = 'none'
        webapp.stopOnDisconnect = False
        webapp.port = args.port
    if args.bind:
        webapp.host = '0.0.0.0'
    if args.dev:
        os.environ['QTWEBENGINE_REMOTE_DEBUGGING'] = '5000'
        webapp.devServerUrl = 'http://127.0.0.1:3000/'
        webapp.jsLogLevel = 'info'

    sys.path.insert(0, '.')  # make sure data sources from .py files in the current folder are found
    source = qmt.dataSourceFromJson(args.source)
    webapp.setupOnlineLoop(source, Demo(source.Ts))

    webapp.run()


if __name__ == '__main__':
    main()
