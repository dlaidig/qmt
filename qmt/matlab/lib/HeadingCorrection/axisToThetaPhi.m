% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [theta,phi] = axisToThetaPhi(j,var)

if(var == 1)   
    theta = acos(j(3));
    phi = atan2(j(2),j(1));    
elseif(var == 2)    
    theta = acos(j(1));
    phi = atan2(j(2),j(3));    
end


end

% if var == 1:
%         theta = np.arccos(j[2])
%         phi = np.arctan2(j[1], j[0])
%     elif var == 2:
%         theta = np.arccos(j[0])
%         phi = np.arctan2(j[1], j[2])