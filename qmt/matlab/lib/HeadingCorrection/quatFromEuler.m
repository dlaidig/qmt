% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = quatFromEuler(angles,convention,intrinsic)

axes = axesFromConvention(convention);

q1 = getQuat(angles(:,1),axes(1,:));
q2 = getQuat(angles(:,2),axes(2,:));
q3 = getQuat(angles(:,3),axes(3,:));

if(intrinsic)
    out = quaternionMultiply(quaternionMultiply(q1,q2),q3);
else
    out = quaternionMultiply(quaternionMultiply(q3,q2),q1);
end
end