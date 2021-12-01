% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = resetAlignment(quaternions,x,xCs,y,yCs,z,zCs,exactAxis)

if(xCs ~= 0)
    x = quaternionRotate(quaternions(xCs),x);
end
if(yCs ~= 0)
    y = quaternionRotate(quaternions(yCs),y);
end

if(zCs ~= 0)
    z = quaternionRotate(quaternions(zCs),z);
end

qSegToGlobal = Quaternion.fromAxes(x, y, z, exactAxis);


end