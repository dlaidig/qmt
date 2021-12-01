% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [q3] = quaternionMultiply (q1,q2)
    [shapeq1,~] = size(q1);
    [shapeq2,~] = size(q2);
    
    if(shapeq1 == shapeq2)
        shape = shapeq1;
    elseif(shapeq1 == 1)
        shape = shapeq2;
    elseif (shapeq2 == 1)
        shape = shapeq1;
    else
        error('Wrong dimension');
    end
    
    
    q3 = zeros(shape,4);
    q3(:,1) = q1(:,1).*q2(:,1) - q1(:,2).*q2(:,2) - q1(:,3).*q2(:,3) - q1(:,4).*q2(:,4);
    q3(:,2) = q1(:,1).*q2(:,2) + q1(:,2).*q2(:,1) + q1(:,3).*q2(:,4) - q1(:,4).*q2(:,3);
    q3(:,3) = q1(:,1).*q2(:,3) - q1(:,2).*q2(:,4) + q1(:,3).*q2(:,1) + q1(:,4).*q2(:,2);
    q3(:,4) = q1(:,1).*q2(:,4) + q1(:,2).*q2(:,3) - q1(:,3).*q2(:,2) + q1(:,4).*q2(:,1);
    
end
