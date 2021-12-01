% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = axesFromConvention(convention)


for i = 1:3
   switch convention(i)
       case 'x'
           out(i,:) = [1 0 0];
       case 'y'
           out(i,:) = [0 1 0];
       case 'z'
           out(i,:) = [0 0 1];
   end
end


end