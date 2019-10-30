function [data, header] = read(fileName, doPrint)
% artia.em.read -- Read files in EM-format
%
% Usage:
%   data = artio_emread(fileName)
%   data = artio_emread(fileName, doPrint)
%   [data, header] = artio_emread(fileName)
%   [data, header] = artio_emread(fileName, doPrint)
%
% Parameters:
%   fileName (str):   
%       File in EM-Format [character vector]
%   doPrint (bool):
%       Whether or not to print info [0/1/true/false], false by default
%
% Returns:
%   data (int, double):   	 
%       Matrix of file contents
%   header (struct):
%       Matlab struct containing header info
%
% See Also:
%   :func:`+artia.+em.write`
%
% Utz H. Ermel 2019

    % Print action?
    if nargin == 1
        doPrint = 0;
    end
    
    % Open file
    fid = fopen(fileName,'r','ieee-le');

    % Read header
    [header, ~, fid] = artia.em.read_header(fid);
    xdim = header.dimX;
    ydim = header.dimY;
    zdim = header.dimZ;

    % Print action
    if doPrint
        fprintf('Reading EM-file: %s with Dimensions:x=%g, y=%g, z=%g\n', fileName, xdim, ydim, zdim);
    end

    % Size
    dsize = xdim * ydim * zdim;

    % Read data
    if header.dataType == 1 % 8-bit integer
        data = fread(fid, dsize,'int8');
    elseif header.dataType == 2 % 16-bit integer
        data = fread(fid, dsize,'int16');
    elseif header.dataType == 4 % 32-bit integer
        data = fread(fid, dsize,'int32');
    elseif header.dataType == 5 % 32-bit float
        data = fread(fid, dsize,'float32');
    elseif header.dataType == 8 % complex 32-bit float
        error('Data type is %d, but complex data types are not supported atm.', header.mode);
    elseif header.dataType == 9 % 64-bit float
        data = fread(fid, dsize,'float64');
    else
        error('Error: Wrong Data Type');
    end
    
    data = reshape(data, [xdim, ydim, zdim]);

    fclose(fid);
end