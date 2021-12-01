% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = randomVector()

out = [randnum randnum randnum];
out = out/norm(out);

end

function out = randnum()

a = rand;
if(a>0.5)
    sign = -1;
else
    sign = 1;
end

out = sign*rand;

end