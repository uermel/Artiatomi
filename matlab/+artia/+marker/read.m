function marker = read(fileName, doPrint)
% artia.marker.read -- Read marker files in extended EM-format (aditional info in
% header and following the data).
%
% Usage:
%    marker = artia.marker.read(fileName)
%
% Inputs:
%    fileName (str):
%       File in EM-Format [character vector]
%    doPrint (bool):
%       Whether or not to print info [0/1/true/false], false by default
%
% Outputs:
%    marker (struct):
%       Matlab structure containing the marker information.
%       1. marker.ali -> The tilt series alignment.
%       2. marker.model -> The real space model.
%
% Author:
%   UE, 2019    

    % Print action?
    if nargin == 1
        doPrint = 0;
    end

    % Read header
    [header, endian] = artia.em.read_header(fileName);
    if ~header.isMarker
        error('Based on the header this EM-file is not a Markerfile. It was probably not created using Clicker or markerWrite.');
    end
    xdim = header.dimX;
    ydim = header.dimY;
    zdim = header.dimZ;

    % Print action
    if doPrint
        fprintf('Reading EM-file: %s with Dimensions:x=%g, y=%g, z=%g\n', fileName, xdim, ydim, zdim);
    end

    % Open file
    fid = fopen(fileName,'r',endian);
    
    % Skip header (512 bytes)
    fread(fid, 512, 'int8');
    
    % Check if this marker file contains the marker model or was not
    % created from Clicker/markerWrite
    % Header + Table + Model
    fileInfo = dir(fileName);
    fileSize = fileInfo.bytes;
    expectedSizeCoords = 512 + 4 * xdim * ydim * zdim + 4 * zdim * 3; 
    if fileSize ~= expectedSizeCoords
        error('Markerfile does not contain 3D marker model. It was probably not created using Clicker or markerWrite.');
    end
    
    % Read data
    I = reshape(fread(fid, xdim * ydim * zdim,'float'), xdim, ydim, zdim);
    coords = reshape(fread(fid, zdim * 3,'float'), 3, zdim);
    
    marker.ali = I;
    marker.model = coords;
    
    % Copy relevant entries from the header
    names = {'isNewHeaderFormat', ...
             'aliScore', ...
             'beamDeclination', ...
             'markerOffset', ...
             'magAnisoFactor', ...
             'magAnisoAngle', ...
             'imageSizeX', ...
             'imageSizeY'};
         %fieldnames(header);
    for i = 1:numel(names)
        marker.(names{i}) = header.(names{i});
    end
    
    % Close file
    fclose(fid);
    
    % Done!
end