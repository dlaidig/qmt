% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function quat = rom_constraint(quat_rel,convention,ranges)

angles = getEulerAngles(quat_rel,convention,true);
axes = axesFromConvention(convention);
angle1 = clip(angles(:,1),ranges(1,1),ranges(1,2));
angle2 = clip(angles(:,2),ranges(2,1),ranges(2,2));
angle3 = clip(angles(:,3),ranges(3,1),ranges(3,2));

quat_rot1 = getQuat(angle1,axes(1,:));
quat_rot2 = getQuat(angle2,axes(2,:));
quat_rot3 = getQuat(angle3,axes(3,:));

quat = quaternionMultiply(quaternionMultiply(quat_rot1,quat_rot2),quat_rot3);

end