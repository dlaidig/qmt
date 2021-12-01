% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = axisFromThetaPhi(theta,phi,var)

if(var == 1)
    out = [sin(theta)*cos(phi) sin(theta)*sin(phi) cos(theta)];
elseif(var == 2)
    out= [cos(theta) sin(theta)*sin(phi) sin(theta)*cos(phi)];
else
    error('Wrong variant');
end


end