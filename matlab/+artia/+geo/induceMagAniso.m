function distorted = induceMagAniso(coords, amount, angle, dims)
% artia.geo.induceMagAniso applies an affine transformation to input coordinates to
% induce magnification anisotropy.
%
% Parameters:
%   coords (double[Nx2]):
%       The input coordinates.
%   amount (double):
%       The anisotropy factor (major axis / minor axis, i.e. ellipticity).
%   angle (double):
%       The angle of the minor axis to the x-axis (degrees).
%   dims (double[2])
%       The image dimensions.
%
% Returns:
%   distorted (double[Nx2]):
%       The distorted coordinates.
%
% Author:
%   UE, 2019
%
    % Get distortion matrices -- these are the matrices for correcting
    % distortion, so inversion of the stretch is necessary later on
    [Sh2, Ro2, St, Ro1, Sh1] = artia.geo.distortionMatrices(angle, amount, dims(1), dims(2));
    
    % Transpose if necessary
    if size(coords, 2) ~= 2
        coords = coords';
    end
    
    % Init output
    distorted = zeros(size(coords));
    
    % Invert the stretch to induce distortion
    St(1, 1) = 1/St(1,1);
    M = Sh2 * Ro2 * St * Ro1 * Sh1;
    
    % Apply the distortion
    for i = 1:size(coords, 1)
        vec = [coords(i, 1), coords(i, 2), 1]';
        
        res = M * vec;
        
        distorted(i, 1) = res(1);
        distorted(i, 2) = res(2);
    end
end