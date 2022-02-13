# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import numpy as np
import qmt
from scipy import optimize

from qmt.utils.plot import AutoFigure


def calibrateMagnetometerSimple(gyr, acc, mag, Ts, targetNorm=49.8, debug=False, plot=False):
    """
    Performs a simple magnetometer calibration.

    The input data should consist of slow rotations in arbitrary directions without any translation and far away from
    potential magnetic disturbances. Calibration is performed by optimizing a cost function that tries to ensure that
    the resulting magnetometer measurements have constant norm and a constant dip angle.

    To apply the calibration parameters::

        mag_calibrated = gain * mag - bias

    :param gyr: Nx3 array with gyroscope measurements [rad/s]
    :param acc: Nx3 array with accelerometer measurements [m/s^2]
    :param mag: Nx3 array with magnetometer measurements or None [any unit]
    :param Ts: sampling time in seconds
    :param targetNorm: norm of the magnetic field in the calibration[µT]
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **gain**: scaling factor, array with shape (3,)
        - **bias**: offset, array with shape (3,)
        - **success**: returns False if the data indicates that calibration failed
        - **debugData**: debug: dict with debug values (only if debug==True)
    """
    gyr = np.ascontiguousarray(gyr, float)
    acc = np.ascontiguousarray(acc, float)
    mag = np.ascontiguousarray(mag, float)

    # calculate initial bias estimate based on the range of measured values and gain estimate based on mean norm
    initBias = (np.min(mag, axis=0) + np.max(mag, axis=0)) / 2
    initGain = 1 / np.mean(qmt.vecnorm(mag-initBias))
    initBias = initBias*initGain
    init = np.concatenate([np.ones(3), initBias])

    # calculate magnetometer-free orientation that is need to calculate dip angle
    params = dict(Ts=Ts, restBiasEstEnabled=False, motionBiasEstEnabled=False, magDistRejectionEnabled=False)
    quat = qmt.oriEstOfflineVQF(gyr, acc, params=params)

    def costFn(x, returnDict=False):
        gain = x[:3]
        bias = x[3:]
        mag_calibrated = initGain * gain * mag - bias

        norm = qmt.vecnorm(mag_calibrated)
        normCost = np.sqrt(np.mean((1 - norm) ** 2))
        normCost = normCost / np.mean(np.abs(gain))

        magE = qmt.normalized(qmt.rotate(quat, mag_calibrated))
        dip = np.arccos(magE[:, 2])
        dipCost = np.var(dip)

        cost = normCost + dipCost
        if returnDict:
            return dict(
                gain=gain*initGain*targetNorm,
                bias=bias*targetNorm,
                internalGain=gain,
                internalBias=bias,
                normCost=normCost,
                dipCost=dipCost,
                cost=cost,
            )
        return cost

    opt_res = optimize.minimize(costFn, init, method='BFGS')
    result = costFn(opt_res.x, returnDict=True)

    mag_calib = result['gain']*mag - result['bias']

    max_val = np.max(mag_calib/targetNorm, axis=0)
    min_val = np.max(-mag_calib/targetNorm, axis=0)
    ranges = np.minimum(max_val, min_val)
    success = np.all(ranges > 0.3)

    if debug or plot:
        g = 1/initGain/targetNorm  # gain for "mag_calibrated = mag" in cost function
        debugData = dict(
            before=costFn(np.array([g, g, g, 0, 0, 0], float), returnDict=True),
            init=costFn(init, returnDict=True),
            result=result,
            opt_res=opt_res,
            success=success,
            ranges=ranges,
            mag=mag,
            mag_norm=qmt.vecnorm(mag),
            mag_dip=np.pi/2-np.arccos(qmt.normalized(qmt.rotate(quat, mag))[:, 2]),
            mag_calib=mag_calib,
            mag_calib_norm=qmt.vecnorm(mag_calib),
            mag_calib_dip=np.pi/2-np.arccos(qmt.normalized(qmt.rotate(quat, mag_calib))[:, 2]),
            targetNorm=targetNorm,
        )
        if plot:
            calibrateMagnetometerSimple_debugPlot(debugData, plot)
        if debug:
            return result['gain'], result['bias'], success, debugData

    return result['gain'], result['bias'], success


def calibrateMagnetometerSimple_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        axes = fig.subplots(2, 1, sharex=True)
        fig.suptitle(AutoFigure.title('calibrateMagnetometerSimple'))

        scale = np.mean(debug['mag_calib_norm']) / np.mean(debug['mag_norm'])
        g, b = debug['result']['gain'], debug['result']['bias']

        axes[0].axhspan(0.95*debug['targetNorm'], 1.05*debug['targetNorm'], color='C2', alpha=0.3)
        axes[0].plot(scale*debug['mag_norm'], label=f'before*{scale:.3f}')
        axes[0].plot(debug['mag_calib_norm'], label='after')
        axes[0].set_title(f'gain: [{g[0]:.3f} {g[1]:.3f} {g[2]:.3f}], bias: [{b[0]:.3f} {b[1]:.3f} {b[2]:.3f}]\n'
                          f'cost: {debug["result"]["cost"]:.4f}, success: {debug["success"]} '
                          f'({np.min(debug["ranges"]):.2f} {">" if debug["success"] else "<="} 0.3)\n\n'
                          f'mag norm, {debug["mag_norm"].shape}, norm cost: {debug["result"]["normCost"]:.4f}')

        meanDip = np.rad2deg(np.mean(debug['mag_calib_dip']))
        axes[1].axhspan(meanDip - 10, meanDip + 10, color='C2', alpha=0.3)
        axes[1].plot(np.rad2deg(debug['mag_dip']), label='before')
        axes[1].plot(np.rad2deg(debug['mag_calib_dip']), label='after')
        axes[1].set_title(f'mag dip [°], {debug["mag_norm"].shape}, dip cost: {debug["result"]["dipCost"]:.4f}')

        for ax in axes:
            ax.grid()
            ax.legend()

        fig.tight_layout()
