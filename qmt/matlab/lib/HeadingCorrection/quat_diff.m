% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = quat_diff(quat1,quat2)

rel_quat = relativeQuaternion(quat1,quat2);
angle = angleFromQuat(rel_quat);

out = angle;


end