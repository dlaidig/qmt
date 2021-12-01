function [R,q] = body2geodetic(gb, mb)
% Returns the rotation matrix R and corresponding quaternion that will
% rotate a vector expressed in the body frame to the geodetic frame where
% x points to north, z points up and y points west.
% Hence R*gb will be [0,0,g], with g positive, since the effect of the 
% gravitational field is indistinguishable from an acceleration upwards.
%
% Input
%    gb   ->  gravitational acceleration vector in the body frame. The 
%             vector points upwards
%    mb   ->  gradient of magnetic field in the body frame. The horizontal
%             component of this points north.
% Output
%    R    <-  Rotation vector
%    q    <-  quaternian
gn = gb/norm(gb);
z = gn;
x = mb - (mb'*gn)*gn;
x = x/norm(x);
y = cross(z,x);
RT = [x, y, z];
R = RT';
q = mat2quat(R);