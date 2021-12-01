function q = mat2quat(R)

if size(R,1) == 3 && size(R,2) == 3
    tr = trace(R);
    if tr > 0
        S = sqrt(tr+1)*2;
        q0 = S/4;
        qx = (R(3,2) - R(2,3))/S;
        qy = (R(1,3) - R(3,1))/S;
        qz = (R(2,1) - R(1,2))/S;
    elseif R(1,1) > R(2,2) && R(1,1) > R(3,3)
        S = sqrt(1 + R(1,1) - R(2,2) - R(3,3))*2;
        q0 = (R(3,2) - R(2,3))/S;
        qx = S/4;
        qy = (R(1,2) + R(2,1))/S;
        qz = (R(1,3) + R(3,1))/S;
    elseif R(2,2) > R(3,3)
        S = sqrt(1 + R(2,2) - R(1,1) - R(3,3))*2;
        q0 = (R(1,3) - R(3,1))/S;
        qx = (R(1,2) + R(2,1))/S;
        qy = S/4;
        qz = (R(2,3) + R(3,2))/S;
    else
        S = sqrt(1 + R(3,3) - R(1,1) - R(2,2))*2;
        q0 = (R(2,1) - R(1,2))/S;
        qx = (R(1,3) + R(3,1))/S;
        qy = (R(2,3) + R(3,2))/S;
        qz = S/4;
    end
else
    error('Input rotation matrix must be 3x3.');
end

q = [q0 qx qy qz]';
q = q/norm(q);