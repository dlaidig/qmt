function R = quat2mat(q)

q = q/norm(q);
if length(q) == 4
    R = [2*q(1)^2+2*q(2)^2-1 2*q(2)*q(3)-2*q(1)*q(4) 2*q(2)*q(4)+2*q(1)*q(3);
         2*q(2)*q(3)+2*q(1)*q(4) 2*q(1)^2+2*q(3)^2-1 2*q(3)*q(4)-2*q(1)*q(2);
         2*q(2)*q(4)-2*q(1)*q(3) 2*q(3)*q(4)+2*q(1)*q(2) 2*q(1)^2+2*q(4)^2-1];
else
    error('Input quaternion must be of length 4.')
end