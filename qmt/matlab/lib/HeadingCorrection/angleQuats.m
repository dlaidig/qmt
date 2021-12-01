% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function angle = angleQuats(q1,q2)

q_rel = relativeQuaternion(q1,q2);
angle = 2*acos(clip(q_rel(:,1),-1,1));

end