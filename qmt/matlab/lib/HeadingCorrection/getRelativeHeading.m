% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function relativeHeading = getRelativeHeading(q1,q2,varargin)

degrees = false;
if(nargin>2)
    degrees = true;
end

[steps,~] = size(q1);

for i = 1:steps
    relativeHeading(i) = getHeading(q2(i,:)) - getHeading(q1(i,:));
    relativeHeading(i) = limitHeading(relativeHeading(i));    
    if(degrees)
        relativeHeading(i) = relativeHeading(i) * 180 / pi;
    end    
    relativeHeading(i) = round(relativeHeading(i),5,'significant');
end


end


function heading = limitHeading(heading)

while(heading>pi)
    heading = heading - 2*pi;
end
while(heading < -pi)
    heading = heading + 2*pi;
end

end