% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [quat,phi] = quatProj(quat1,quat2,axis)

q_rel = relativeQuaternion(quat1,quat2);

phi = 2*atan2(dot(q_rel(2:4),axis),q_rel(1));

quat = getQuat(phi,axis);

end