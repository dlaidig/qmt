% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function gyr = gyrFromQuat(quat,rate)

N = size(quat,1);

gyr=zeros(N,3);

dq = relativeQuaternion(quat(1:end-1,:),quat(2:end,:));
dq(dq(:,1) < 0) = -dq(dq(:,1) < 0);
angle = 2*acos(clip(dq(:,1),-1,1));
axis = zeros(N-1,3);

nonzero = angle>eps;
axis(nonzero,:) = dq(nonzero,2:end)./vecnorm2(dq(nonzero,2:end));
gyr(2:end,:) = angle .* axis * rate;
end