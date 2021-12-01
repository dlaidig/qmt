% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [heading,inclination,axis_incl] = getHeading(quat,varargin)

 x = quat(:,2);
 y = quat(:,3);
 z = quat(:,4);
 w = quat(:,1);
 axis = [0 0 1];
 heading = 2*atan2([x y z] * axis',w);

 q_rest = quaternionMultiply(quaternionInvert(getQuat(heading,[0 0 1])),quat);
 inclination = 2*acos(q_rest(:,1));
 
 axis_incl = axisFromQuat(q_rest);
 
% x_global = quaternionRotate(quat, [1,0,0]);
% heading = atan2(x_global(:,1), x_global(:,2));


for k = 1:length(heading)
while(heading(k) < 0)
    heading(k) = heading(k) + 2*pi;
end
end
if(nargin>1)
    heading = heading * 180 / pi;
end

end