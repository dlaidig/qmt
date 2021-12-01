% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [q,data] = estimateQuat_OEI(data,useMag,varargin)

if(isfield(data(1),'time'))
    rate = (data(1).time(2) - data(1).time(1))^-1;
    steps = length(data(1).time);
else
    rate = (data(1).t(2) - data(1).t(1))^-1;
    steps = length(data(1).t);
end
nrOfSegments = length(data);
%qInit = [0.5 0.5 0.5 0.5];
qInit = [1 0 0 0];

if nargin > 2
    qInit = varargin{1};
end

if(~useMag)
    fprintf('OriEstImu: no magnetometer\n');
else
    fprintf('OriEstImu: with magnetometer\n');
end



for i = 1:nrOfSegments
    if(~useMag)
        data(i).mag = zeros(steps,3);
        data(i).quat = estimateQuaternion(qInit,data(i).acc, data(i).gyr, data(i).mag, rate, 1, 300000, 0, 1);
    else
        data(i).quat = estimateQuaternion(qInit,data(i).acc, data(i).gyr, data(i).mag, rate, 1,7,0,1);
    end
    q = data(i).quat;
end


end

function [quat] = estimateQuaternion(qInit ,acc, gyr, mag, rate, tauAcc, tauMag, zeta, accRating)
N = size(acc, 1);
state = zeros(1, 18);
state(1:4) = qInit;
for i=1:N
    [state, errorAngleIncl, errorAngleAzi] = CSG_OriEst_IMU(state, acc(i,:), gyr(i,:), mag(i,:), rate, tauAcc, tauMag, zeta, accRating);
    quat(i,:) = state(1:4);
end
end