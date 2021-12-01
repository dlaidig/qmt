% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = getQuat(angle,axis)

if(norm(axis) < eps)
    n = length(angle);
    out = [1 0 0 0];
    out = repmat([1 0 0 0],n,1);
    return
end
for i = 1:size(axis,1)
   axis(i,:) = axis(i,:)/norm(axis(i,:)); 
end


[rows,cols] = size(angle);
if(cols>rows)
    angle_new = angle';
else
    angle_new = angle;
end
out = [cos(angle_new/2),sin(angle_new/2) * axis];
end