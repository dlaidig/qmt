% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function headingError12 = readHeadingError(data,data_dist,seg1,seg2)

relativeHeading12_unDist = getRelativeHeading(data.(seg1).quat,data.(seg2).quat);
relativeHeading12_dist = getRelativeHeading(data_dist.(seg1).quat,data_dist.(seg2).quat);

headingError12 = relativeHeading12_dist - relativeHeading12_unDist;
headingError12 = limitHeading(headingError12);
end


function heading = limitHeading(heading)

for i = 1 : length(heading)
    while(heading(i) < -pi)
        heading(i) = heading(i) + 2*pi;
    end
    while(heading(i) > pi)
        heading(i) = heading(i) - 2*pi;
    end
end
end