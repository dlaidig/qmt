% SPDX-FileCopyrightText: 2021 Thomas Seel <seel@control.tu-berlin.de>
% SPDX-FileCopyrightText: 2021 Stefan Ruppin <ruppin@control.tu-berlin.de>
%
% SPDX-License-Identifier: MIT

% ISF Computer Exercise 3
function [new_estimation_state, errorAngleIncl, errorAngleAzi] = CSG_OriEst_IMU(estimation_state, Acc, Gyro, Mag, sample_freq, TauAcc, TauMag, Zeta, useMeasRating)
    % #############################################################################################################################################################
    % IMU ORIENTATION ESTIMATION
    % Developed at Control Systems Group, TU Berlin
    % (http://www.control.tu-berlin.de)
    % Algorithm design: Thomas Seel
    % Code design: Stefan Ruppin
    % Further reading: paper to be submitted...

    % Contact: {seel,ruppin}@control.tu-berlin.de
    % 
    % All units are SI units.
    %   Inputs:
    %    estimation_state:  state of the estimation before the update
    %                       size of estimation_state [1 18]
    %                       estimation_state = [quaternion, estimatedBias, validAccDataCount, ratingWindow]
    %                       length(quaternion)          == 4
    %                       length(estimatedBias)       == 4
    %                       length(validAccDataCount)   == 1
    %                       length(ratingWindow)        == windowLength
    %
    %   Acc:                measurement of the accelerometer [x,y,z][m/s^2]
    %                       size == [1 3]
    %
    %   Gyro:               measurement of the gyro          [x,y,z][rad/s]
    %                       size == [1 3]
    %
    %   Mag:                measurement of the magnetometer  [x,y,z][any unit] 
    %                       size == [1 3]
    %
    %   sample_freq:        sample frequency of measurement data
    %
    %   TauAcc, TauMag:     time constants for correction (50% time) [must be >0]
    %                       [in seconds]                       
    %
    %   Zeta:               bias estimation strength [no unit]
    %
    %   useMeasRating:      0: do not use raw rating of accelerometer
    %                       1: use raw rating of acclerometer
    %                       Recommended: 1
    %
    %   Outputs:
    %    new_estimation_state:
    %                       new state of the estimation after the update
    %                       size of new_estimation_state [1 18]
    %                       new_estimation_state    = [quaternion, estimatedBias, validAccDataCount, ratingWindow]
    %                       length(quaternion)          == 4
    %                       length(estimatedBias)       == 4
    %                       length(validAccDataCount)   == 1
    %                       length(ratingWindow)        == windowLength(default:10)
    %
    %   errorAngleIncl:     error angle of inclination [scalar]
    %
    %   errorAngleAzi:      error angle of azimuth [scalar]
    % 
    %  How to use this filter:
    %     Frist, choose a start quaternion[4] and an initial bias[3] to begin the estimation with.
    %     Please normalize the start quaternion!
    %     With these start values create a state object: 
    %     state = zeros(1,4+3+winLength) where the default winLength = 10.
    %     
    %     Choose for TauAcc the time that should pass in seconds before the error of inclination is halved.
    %     Analog for TauMag.
    %     Choose for Zeta a value >= 0. This value will define your overshooting. Zeta = 0 means no bias estimation will be done.
    %     Zeta > 0 means that a bias will be estimated. The higher Zeta is chosen, the more overshooting will be present when the reference jumps.
    %     Recommended: Zeta = [1..2]   
    % #############################################################################################################################################################
    % #### Init declarations #####
    zeroEps                         = 1D-6;                                                            % treshold for if zero statementes (for debug)
    if (TauAcc == 0 || TauMag == 0)                                                                     % check taus!=0 to prevent division by zero
        disp('Error: TauMag or TauAcc is zero!');
        return;
    end
    windowLength                    = 10;                                                               % window length for rating
    if ( sum( size(estimation_state) == [1, 4+3+1+windowLength])  ~= 2)                                 % if given state is not valid, assign default
        estimation_state = zeros([1, 4+3+1+windowLength]);
        estimation_state(1:4)       = [0 0 0 1];                                                        % define a deafult quaternion if noone is given
    end
 
    accmeas                         = [Acc(1)  Acc(2)  Acc(3)];                                         % make sure that input is row vector
    gyromeas                        = [Gyro(1) Gyro(2) Gyro(3)];                                        % make sure that input is row vector
    magmeas                         = [Mag(1)  Mag(2)  Mag(3)];                                         % make sure that input is row vector
    bias                            = estimation_state(5:7);
    q_gyro                          = estimation_state(1:4);                                            % set start quaternion for gyro-based prediction
    validAccDataCount               = estimation_state(8);                                              % counter for samples with valid accelemeter data
    window                          = estimation_state(9:9+windowLength-1);
 
    errorAngleIncl                  = 0;                                                                % set default for output errorAngleIncl
    errorAngleAzi                   = 0;                                                                % set default for output errorAngleAzi
 
    if (abs(norm(q_gyro)-1)>zeroEps)                                                                    % check for error (optional)
        disp('input q is not a unit quaternion'); end                                                   % "
 
    gyromeas                        = gyromeas + bias;                                                  % correct gyro bias
 
    % #############################################################################################################################################################
    % Rate raw values for accelerometer and adjust correction gain based on it
    correctionRating                = 1;                                                                % default:no down rating of correction gain
    if (useMeasRating > 0)                                                                              % check if rating is enabled
        window(1:windowLength-1)    = window(2:windowLength);                                           % shift window/buffer one sample up
        window(windowLength)        = abs(norm(accmeas)-9.81);                                          % add new sample to buffer
        if validAccDataCount > 0
            correctionRating            = 1/(1+max(window)*useMeasRating);                                  % calculate correction rating (1: normal thrust, <1 down rating)
        end
    end
 
    % ##############################################################################################################################################################
    % #### Initial estimation ####
    % after starting the algorithm, the time constants are set to small
    % value to ensure fast convergence of the estimation
    % added 09/30/2015
    if (validAccDataCount+1)/sample_freq/2 < TauAcc/2                                                   % start bias estimation after TauAcc/2 is reached
        Zeta = 0;                                                                                       % to prevent estimating the bias out of the error due 
    end                                                                                                 % to the initial orientation
    TauAcc = min((validAccDataCount+1)/sample_freq/2,TauAcc);                                           % Speed-up acceleromter-based correction in the first samples
    TauMag = min((validAccDataCount+1)/sample_freq/2,TauMag);                                           % Speed-up magnetometer-based correction in the first samples
 
    % #############################################################################################################################################################
    % #### Gyro-based prediction #####
    % in: q ( already in q_gyro)  out: q_gyro
    if (gyromeas(1) ~= 0 || gyromeas(2) ~= 0 || gyromeas(3) ~= 0)                                       % check gyro measurement to be valid (!=0)
        gyro_norm                   = norm(gyromeas);                                                   % get norm to calculate prediction angle
        prediction_ang              = gyro_norm / (sample_freq);                                        % calculate angle to correct in angle/axis
        dq_gyro                     = [cos(prediction_ang/2), ...                                       % specify quaternion for gyro correctin
                                       sin(prediction_ang/2)*(gyromeas/gyro_norm)];                     % "
        q_gyro                      = quaternionMultiply(q_gyro, dq_gyro);                              % correct initial quaternion by dq_gyro in IMU frame!
 
        if (abs(norm(dq_gyro)-1)>zeroEps)                                                               % check for error (optional)
            disp('dq_gyro is not a unit quaternion'); end                                               % "
        if (abs(norm(q_gyro)-1)>zeroEps)                                                                % "
            disp('q_gyro is not a unit quaternion'); end                                                % "
    end
 
    % #############################################################################################################################################################
    % #### Accelerometer-based correction ####
    % in: q_gyro   out: q_gyro_acc
    q_gyro_acc  = q_gyro;                                                                               % define to get an output even if accmeas is zero
    gravref_fixedframe              = [0 0 1];                                                          % define gravitation reference (vertical) in fixed frame
    gravref_imuframe                = quaternionCoordTransform(q_gyro,[0,gravref_fixedframe]);          % transform gravitation reference into IMU frame
 
    if (accmeas(1) ~= 0 || accmeas(2) ~= 0 || accmeas(3) ~= 0)                                          % check accmeas to be valid
        validAccDataCount = validAccDataCount+1;                                                        % count update calls of the algorithm with valid acceleromter data
        accmeas = accmeas/norm(accmeas);                                                                % normalize accmeas
        if (abs(accmeas*gravref_imuframe(2:4)')< 1)                                                     % perform correction only if accmeas and gravref_imuframe do NOT coincide
            errorAngleIncl          = acos(accmeas*gravref_imuframe(2:4)');                             % calculate error between reference and measurment          
            kp_acc                  = correctionRating ...                                              % calculate correction gain kp_acc from time constant
                                      * (1 - 1.4*TauAcc*sample_freq/(1.4*TauAcc*sample_freq+1));        % "
            correction_ang          = kp_acc * errorAngleIncl;                                          % calculate angle for correction
            correctionaxis_imuframe = cross(accmeas, gravref_imuframe(2:4));                            % rotation axis of correction is perpendicular to measurement and reference
            correctionaxis_imuframe = correctionaxis_imuframe/norm(correctionaxis_imuframe);            % normalize rotation axis
            dq_acc                  = [cos(correction_ang/2), ...                                       % build correction quaternion from axis and angle
                                       sin(correction_ang/2)*correctionaxis_imuframe];                  % "
            q_gyro_acc              = quaternionMultiply(q_gyro,dq_acc);                                % correct gyro corrected quaternion by acc correction
            if kp_acc < 1
                ki_acc                  = (Zeta^2/160)*1.4*sample_freq *kp_acc^2/(1-kp_acc);            % calculate ki_acc for bias estimation
                bias                    = bias + ...                                                    % estimate bias from accelerometer correction
                                          ki_acc * errorAngleIncl*correctionaxis_imuframe;              % "
            end
            if (abs(norm(gravref_imuframe)-1)>zeroEps)                                                  % check for error (optional)
                disp('gravref_imuframe is not a unit quaternion'); end                                  % "
            if (abs(norm(dq_acc)-1)>zeroEps)                                                            % "
                disp('dq_acc is not a unit quaternion'); end                                            % "
            if (abs(norm(q_gyro_acc)-1)>zeroEps)                                                        % "
                disp('q_gyro_acc is not a unit quaternion'); end;                                       % "
        end
    end
    % #############################################################################################################################################################
    % #### Magnetometer-based correction ####
    % in: q_gyro_acc   out: q_gyro_acc_mag
    q_gyro_acc_mag                  = q_gyro_acc;                                                       % define to get an output even if input is zero
    if (magmeas(1) ~= 0 || magmeas(2) ~= 0 || magmeas(3) ~= 0)                                          % check magmeas to be valid
        magref_fixedframe           = [0 1 0];                                                          % define magnetic field reference in fixed frame (choose [+/-1 0 0] or [0 +/-1 0] to define which fixed-frame coordinate axis points north/south)
        magref_imuframe             = quaternionCoordTransform(q_gyro_acc, [0,magref_fixedframe]);      % transform magnetic field reference into IMU frame
        gravref_imuframe            = quaternionCoordTransform(q_gyro_acc, [0, gravref_fixedframe]);    % recalculate IMU frame coordinates of vertical axis (might be skipped)
 
        if (abs(magmeas*gravref_imuframe(2:4)')<norm(magmeas))                                          % if measured magnetic field is vertical, perform NO correction
            magmeas_projected       = (magmeas' - (gravref_imuframe(2:4)*magmeas') ...                  % projection of magnetic measurement into horizontal plane
                                                   *gravref_imuframe(2:4)')';                           % "
            magmeas_projected       = magmeas_projected/norm(magmeas_projected);                        % normalize projection
            if (abs(magmeas_projected*magref_imuframe(2:4)')<1)                                         % if projected measurement and reference agree, perform NO correction
                errorAngleAzi       = acos(magmeas_projected*magref_imuframe(2:4)');                    % calculate error angle between reference and measurment
                kp_mag              = (1 - 1.4*TauMag*sample_freq/(1.4*TauMag*sample_freq+1));          % calculate correction gain  kp_mag from time constant
                correction_ang      = kp_mag*errorAngleAzi;                                             % calculate angle for correction
                correctionaxis_imuframe     = cross(magmeas_projected,magref_imuframe(2:4));            % rotation axis of correction is perpendicular to measurement and reference
                correctionaxis_imuframe     = correctionaxis_imuframe/norm(correctionaxis_imuframe);    % normalize rotation axis
                dq_mag              = [cos(correction_ang/2), ...                                       % build correction quaternion from axis and angle
                                       sin(correction_ang/2)*correctionaxis_imuframe];                  % "
                q_gyro_acc_mag      = quaternionMultiply(q_gyro_acc,dq_mag);                            % correct gyro&acc-corrected quaternion by mag correction
                if kp_mag < 1
                    ki_mag              = (Zeta^2/160)*1.4*sample_freq *kp_mag^2/(1-kp_mag);            % calculate ki_mag for bias estimation
                    bias                = bias + ...                                                    % estimate bias from magnetometer correction
                                          ki_mag * errorAngleAzi * correctionaxis_imuframe;             % "
                end
                                   
                if (abs(norm(dq_mag)-1)>zeroEps)                                                        % check for error (optional)
                disp('dq_mag is not a unit quaternion'); end                                            % "
            end
        end
    end
 
    if (abs(norm(q_gyro_acc_mag)-1)>eps)                                                                % check final quaternion to be unit (could be false due to numerics)
        q_gyro_acc_mag= q_gyro_acc_mag/norm(q_gyro_acc_mag);                                            % if not unit quaternion, normalize it to prevent error porpagation
    end                                                                                                 % "
     
%     if q_gyro_acc_mag(1) < 0                                                                            % make sure the first entry of the quaternion is always positive (for comfort)
%         q_gyro_acc_mag = -1*q_gyro_acc_mag;                                                             % reason: there is an ambiguity in the quaternion: -q = q for orientation
%     end                                                                                                 % "
     
    new_estimation_state            = [ q_gyro_acc_mag, bias, validAccDataCount, window];               % return new state of estimation after update
end
function [q3] = quaternionCoordTransform (q1,q2)
   % This function will do a rotation with two quaternions
   % Result: q1' * q2 * q1
   % The result will always be a quaternion [x x x x]
   q1Inv = quaternionInvert(q1);
   q3 = quaternionMultiply(q1Inv,q2);
   q3 = quaternionMultiply(q3,q1);
end
