function R = rotMat(x)

if length(x) == 4 && numel(x) == 4
    %Calculates the rotation matrix R from the quaternion q
    x = x/norm(x);
    q0 = x(1); q1 = x(2); q2 = x(3); q3 = x(4);

    R = [2*(q0^2+q1^2) - 1  2*(q1*q2-q0*q3)    2*(q1*q3+q0*q2);
         2*(q1*q2+q0*q3)    2*(q0^2+q2^2) - 1  2*(q2*q3-q0*q1);
         2*(q1*q3-q0*q2)    2*(q2*q3+q0*q1)    2*(q0^2+q3^2) - 1];
elseif length(x) == 3 && numel(x) == 3
    %Calculated the rotation matrix R from the Euler angles r, p, h
    r = x(1); % Roll 
    p = x(2); % Pitch 
    y = x(3); % Yaw
    
    R = [cos(y)*cos(p)                       sin(y)*sin(p)                      -sin(p);
        -sin(y)*cos(p)+cos(y)*sin(p)*sin(r)  cos(y)*cos(r)+sin(y)*sin(p)*sin(r)  cos(p)*sin(r);
         sin(y)*sin(p)+cos(y)*sin(p)*cos(r) -cos(y)*sin(r)+sin(y)*sin(p)*cos(r)  cos(p)*cos(r)];    
else
    error('Input has to be either a 4-dim quaternion or 3-dim Euler angles.')
end