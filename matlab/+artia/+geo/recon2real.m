function [model] = recon2real(xyz, reconDim, voxelSize, volumeShifts)
% artia.geo.recon2real transforms binned, shifted reconstruction
% coordinates to global coordinates.
%
% Parameters:
%   xyz (double):
%       The reconstruction volume coordinates.
%   reconDim (double[3]):
%       The reconstruction dimensions.
%   voxelSize (double):
%       The voxel size of the reconstruction.
%   volumeShifts (double[3]):
%       The volume shifts applied during reconstruction.
%
% Returns: 
%   model:
%       The real space model.
%
% Author:
%   UE, 2019
%

    % Binned, shifted tomogram coordinates to unbinned, unshifted coordinates
    %  --> Center points and remove binning
    %  --> Need to add -0.5 because: -1 for C++ -> Matlab, +0.5 for middle
    %      of voxel
    tx = (xyz(1,:) - 0.5 - reconDim(1)/2) * voxelSize - volumeShifts(1);
    ty = (xyz(2,:) - 0.5 - reconDim(2)/2) * voxelSize - volumeShifts(2);
    tz = (xyz(3,:) - 0.5 - reconDim(3)/2) * voxelSize - volumeShifts(3);

    
    % Rotate 90 degrees and mirror along new y-axis to transform to world
    % coordinates   
    x = ty;
    y = -tx;
    z = tz;
    
    % To marker struct
    model = [x; y; z];
    
    % Done!
end

