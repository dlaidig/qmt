function [x,r] = vector2spherical(v)

if numel(v) == 3
    x(1,1) = asin(v(3));
    x(2,1) = atan2(v(2),v(1));
    r = norm(v,2);
else
    error('Vector must be 3D to be converted to spherical coordinates')
end