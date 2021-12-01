% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function q = rotMatToQuat(M)

w = sqrt(1/4*(1+M(1,1)+M(2,2)+M(3,3)));
x = sign(M(3,2)-M(2,3))*sqrt(1/4*(1+M(1,1)-M(2,2)-M(3,3)));
y = sign(M(1,3)-M(3,1))*sqrt(1/4*(1-M(1,1)+M(2,2)-M(3,3)));
z = sign(M(2,1)-M(1,2))*sqrt(1/4*(1-M(1,1)-M(2,2)+M(3,3)));

q = [w x y z];

end