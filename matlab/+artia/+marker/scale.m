function marker = scale(marker, pixelSize, newPixelSize, varargin)
% artia.marker.scale rescales the 3D alignment of tomographic tilt series.
%
%
% Parameters:
%   marker (struct, str): 
%       Markerfile structure or filename of a markerfile 
%   voxelSize (int): 
%       Current voxel size of the marker file
%   newVoxelSize (int): 
%       Voxel size of the output marker
%
% Name Value Pairs:
%   'OutFile' (str, optional): Filename of an output marker file
%
% Returns:
%   marker (struct) - A markerfile structure. 
%
% Author:
%   Utz H. Ermel, 2019

    % Read if necessary
    if ischar(marker)
        marker = artia.marker.read(marker);
    end

    % Specify output file
    p = inputParser;
    paramName = 'OutFile';
    defaultVal = '';
    addParameter(p,paramName,defaultVal);
    parse(p, varargin{:});
    outFile = p.Results.OutFile;
    
    % Scaling factor
    fac = pixelSize/newPixelSize;
    
    % Scale marker
    marker.ali([2:3, 5:6], :, :) = marker.ali([2:3, 5:6], :, :) .* fac;
    marker.model = marker.model .* fac;
    marker.ImSizeX = marker.ImSizeX * fac;
    marker.ImSizeY = marker.ImSizeY * fac;
    
    % Save if necessary
    if ~isempty(outFile)
        artia.marker.write(marker, outFile);
    end
end