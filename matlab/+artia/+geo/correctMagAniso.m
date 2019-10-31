function undistorted = correctMagAniso(coords, amount, angle, dims)
% artia.geo.correctMagAniso applies an affine transformation to input coordinates
% to correct the effect of anisotropic magnification anisotropy.
%
% Parameters:
%   coords (double[Nx2]):
%       The coordinates to be corrected.
%   amount (double):
%       The anisotropy factor (major axis / minor axis, i.e. ellipticity).
%   angle (double):
%       The angle of the minor axis to the x-axis (degrees).
%   dims (double[2])
%       The image dimensions.
% 
% Returns:
%   undistorted (double[Nx2]):
%       The corrected coordinates.
%
% Author:
%   UE, 2019
%
    % Get distortion matrices 
    [Sh2, Ro2, St, Ro1, Sh1] = distortionMatrices(angle, amount, dims(1), dims(2));
    
    % Transpose if necessary
    if size(coords, 2) ~= 2
        coords = coords';
    end
    
    % Init output
    undistorted = zeros(size(coords));
    
    % Distortion matrix
    M = Sh2 * Ro2 * St * Ro1 * Sh1;
    
    % Apply distortion
    for i = 1:size(coords, 1)
        vec = [coords(i, 1), coords(i, 2), 1]';
        
        res = M * vec;
        
        undistorted(i, 1) = res(1);
        undistorted(i, 2) = res(2);
    end
end