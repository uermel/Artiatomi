function angles = matrix2euler(rm)
% artia.geo.matrix2euler computes EmSART euler angles from a rotation
% matrix (phi, theta, psi, ZYZ convention).
%
% Parameters:
%   rm (double[3x3]):
%       The rotation matrix.
%
% Returns:
%   angles (double[3]):
%       [phi, psi, theta], the euler angles.
%
% Author:
%   DCD, UE
%

tol = 1e-4;
if rm(3,3)>1-tol && rm(3,3)<1+tol
    warning('indetermination in defining phi and psi: rotation about z');
    theta = 0;
    phi = -atan2(rm(2, 1), rm(1, 1)) * 180 / pi;
    psi = 0;
else
    theta = acos(rm(3, 3));
    phi = atan2(rm(1, 3), rm(2, 3));
    psi = atan2(rm(3, 1), -rm(3, 2));


    theta = theta * 180 / pi;
    phi = phi * 180 / pi;
    psi = psi * 180 / pi;
end

angles = [phi, psi, theta];