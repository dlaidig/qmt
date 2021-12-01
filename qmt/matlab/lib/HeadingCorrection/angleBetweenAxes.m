% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function angle = angleBetweenAxes(axis1,axis2)
angle = acos(dot(axis1,axis2)/(norm(axis1)*norm(axis2)));
cp = cross(axis1, axis2);
if (dot([0 0 1], cp) < 0)
    angle = -angle;
end


end

