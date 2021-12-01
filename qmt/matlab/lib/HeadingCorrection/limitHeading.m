% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function heading = limitHeading(heading)
for i = 1:length(heading)   
    while(heading(i) <= -pi)
        heading(i) = heading(i) + 2*pi;
    end
    while(heading(i) > pi)
        heading(i) = heading(i) - 2*pi;
    end
end
end