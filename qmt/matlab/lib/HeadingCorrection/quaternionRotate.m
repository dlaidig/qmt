% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function v = quaternionRotate (q, vec)
   % This function will rotate the vectors v (1x3 or Nx3) by the quaternions
   % q (1x4 or Nx4)
   % Result: q * [0,v] * q'
   % The result will always be a vector (Nx3)
   
   if(size(q,1) == 4 && size(q,2) ~= 4)
       q = q';
   end
   
   if(size(q,1) > size(vec,1) && size(vec,1) == 1)
       vector_extender = size(q,1);
   else
       vector_extender = 1;
   end
   
   qInv = quaternionInvert(q);
   qv = quaternionMultiply(quaternionMultiply(q, repmat([zeros(size(vec, 1), 1), vec],vector_extender,1)), qInv);
   v = qv(:, 2:4);
end
