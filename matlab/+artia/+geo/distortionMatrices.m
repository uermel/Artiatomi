function [Sh2, Ro2, St, Ro1, Sh1] = distortionMatrices(angle, amount, dimX, dimY)
% artia.geo.distortionMatrices computes the affine transformations
% used for correction or induction of magnification anisotropy.
%
% Parameters:
%   angle (double):
%       The angle of the minor axis to the x-axis (degrees).
%   amount (double):
%       The anisotropy factor (major axis / minor axis, i.e. ellipticity).
%   dimX (double):
%       The image X dimension.
%   dimY (double):
%       Teh image Y dimension.
%
% Returns:
%   Sh2 (double[3x4]):
%       Shift matrix 2 (shift to projection coords).
%   Ro2 (double[3x4]):
%       Rotation matrix 2 (rotate to projection orientation).
%   St (double[3x4]):
%       Stretch matrix (stretch x-axis).
%   Ro1 (double[3x4]):
%       Rotation matrix 1 (rotate minor axis to x-axis).
%   Sh1 (double[3x4]):
%       Shift matrix 1 (shift to global projection coords).
%
% Author:
%   UE, 2019
%

    % "shiftCenter" matrix in C++
    Sh1 = [1,       0,   -dimX/2;
           0,       1,   -dimY/2;
           0,       0,         1];
    
    % "shiftBack" matrix in C++
    Sh2 = [1,       0,     dimX/2;
           0,       1,     dimY/2;
           0,       0,          1];
       
    % "stretch" matrix in C++
    St =  [amount,  0,            0;
           0,       1,            0;
           0,       0,            1];
       
    % "rotMat1" in C++ 
    angle = deg2rad(angle);
    cosa = cos(angle);
    sina = sin(angle);
    Ro1 = [cosa,    -sina,      0;
           sina,     cosa,      0;
           0,           0,      1];
    
    % "rotMat2" in C++
    cosa = cos(-angle);
    sina = sin(-angle);
    Ro2 = [cosa,    -sina,      0;
           sina,     cosa,      0;
           0,           0,      1];

end