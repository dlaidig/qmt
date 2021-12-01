% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [dTheta,dPhi] = axisGradient(theta,phi,var)

if(var == 1)
   dTheta = [cos(theta)*cos(phi) cos(theta)*sin(phi) -sin(theta)];
   dPhi = [-sin(theta)*sin(phi) sin(theta)*cos(phi) 0];
elseif(var ==2)
    dTheta = [-sin(theta) cos(theta)*sin(phi) cos(theta)*cos(phi)];
    dPhi = [0 sin(theta)*cos(phi) -sin(theta)*sin(phi)];
else
    error('Wrong variant');
end


end