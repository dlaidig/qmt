% SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function  [quat2_corr, delta, delta_filt, rating, state_out, debug_data] = headingCorrection(gyr1, gyr2, quat1, quat2, t, joint, jointInfo, estSettings, debug)
    % This function corrects the heading of a kinematic chain of two segments whose orientations are estimated without
    % the use of magnetometers. It corrects the heading of the second segment in a way that its orientation is estimated
    % in the reference of the first segment. It uses kinematic constraints for rotational joints to estimate the heading
    % offset of both segments based on the limited set of possible relative orientations induced by the rotational
    % joint. There are methods for 1D, 2D and 3D joints, based on different constraints.
    %
    % Equivalent Python function: :func:`qmt.headingCorrection`.
    %
    % :param gyr1: Nx3 vector of angular velocities in rad/s of first segment.
    % :param gyr2: Nx3 vector of angular velocities in rad/s of second segment.
    % :param quat1: Nx4 vector of orientation quaternions of first segment.
    % :param quat2: Nx4 vector of orientation quaternions of second segment.
    % :param t: Nx1 vector of the equidistant sampled time signal
    % :param joint:  dofx3 array containing the axes of the joint.
    %
    % :param jointInfo: structure containing additional joint information. only needed for 3D-joints. Fields:
    %
    %               - convention: String of the chosen euler angles convention,
    %                 e.g. 'xyz'
    %               - angle_ranges: 3x2 array with the minimum and maximum
    %                 value for each joint angle in radians
    %
    % :param estSettings: Structure containing the needed settings for the estimation. Fields:
    %
    %               - window_time: width of the window over which the estimation should be performed in seconds
    %               - estimation_rate: rate in Hz at which the estimation should be performed. typically below 1Hz is sufficient
    %               - data_rate: rate at which new data is fed into the estimation. Is used to downsample the data for a faster estimation. Typically values around 5Hz are sufficient
    %               - alignment: String of the chosen type of alignment of the estimation window. It describes how the samples around the  current sample at which the estimation takes place are chosen. Possible values: 'backward', 'center', 'forward'
    %               - enable_stillness: boolean value if the stillness detection should be enabled
    %               - stillness_time: time in seconds which should pass for the stillness detection to be triggered
    %               - stillness_threshold: threshold for the angular velocity under which the body is assumed to be at rest. In rad/s
    %               - tau_delta: time constant for the filter in seconds
    %               - tau_bias: time constant for the filter in seconds
    %               - delta_range: array of values that are supposed to be tested for the method for 3D joints.
    %               - constraint: type of contraint used for the estimation.
    %
    %                   - Constraints for 1D: (proj, euler_1d, euler_2d, icorr).
    %                   - Constraints for 2D: (euler,euler_lin, gyro, combined).
    %                   - Constraints for 3D: (default)
    %
    %               - optimizer_steps: Number of Gauss-Newton steps during optimization.
    %
    % :return:
    %           - quat2_corr: Corrected orientation of the second segment in the reference
    %             frame of the first segment
    %           - delta: value of the estimated heading offset
    %           - delta_filt: value of the filtered heading offset
    %           - rating:     value of the certainty of the estimation
    %           - state:      state of the estimation. 1: regular, 2: startup, 3: stillness

    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'lib', 'HeadingCorrection'))

    clear heading_correction;

    if(nargin<9)
        debug = 0;
    end
    if (nargin<8)
        estSettings = struct.empty();
    end
    if (nargin<7)
        jointInfo = struct.empty();
    end
    if iscell(joint)
        joint = cell2array(joint);
    end
    estSettings = containersmap2Struct(estSettings);
    jointInfo = containersmap2Struct(jointInfo);

    data1.gyr = gyr1;
    data1.quat = quat1;
    data1.time = t;

    data2.gyr = gyr2;
    data2.quat = quat2;
    data2.time = t;

    [quat2_corr, delta, delta_filt, rating ,state_out, debug_data] = heading_correction(data1, data2, joint, jointInfo, estSettings, debug);


end
