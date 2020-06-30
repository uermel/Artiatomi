function [xyz] = real2recon(model, reconDim, voxelSize, volumeShifts)
% artia.geo.real2recon transforms the 3D marker model from global
% coordinates to binned, shifted reconstruction coordinates.
%
% Parameters:
%   model (double[3xM]):
%       The real space model to be converted (M markers).
%   reconDim (double[3]):
%       The reconstruction dimensions.
%   voxelSize (double):
%       The voxel size of the reconstruction.
%   volumeShifts (double[3]):
%       The volume shifts applied during reconstruction.
%
% Returns:
%   xyz (double):
%       The reconstruction volume coordinates.
%
% Author:
%   UE, 2019
%

    % Mirror along y-axis and rotate -90 degrees to transform to recon
    % coordinates   
    mx = -model(2, :);
    my = model(1, :);
    mz = model(3, :);
    
    % Add shifts
    x = (mx + volumeShifts(1))./voxelSize;
    y = (my + volumeShifts(2))./voxelSize;
    z = (mz + volumeShifts(3))./voxelSize;
    
    % Shift Origin, +0.5 to move from center of voxel
    x = x + reconDim(1)/2 + 0.5;
    y = y + reconDim(2)/2 + 0.5;
    z = z + reconDim(3)/2 + 0.5;
    
    % Concat
    xyz = [x; y; z];
    
    % Done!
end

