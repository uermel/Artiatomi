function [data, header] = read(fileName, doPrint)
% artio_mrcread -- Read files in MRC-format
%
%  Usage:
%    data = artio_mrcread(fileName)
%    data = artio_mrcread(fileName, doPrint)
%    [data, header] = artio_mrcread(fileName)
%    [data, header] = artio_mrcread(fileName, doPrint)
%
%  Inputs:
%    fileName    File in MRC-Format 
%    doPrint     Whether or not to print info [0/1/true/false], false by
%                default
%
%  Outputs:
%    data    	 Matrix of file contents
%    header      Matlab struct containing header info
%
%  See Also
%    artio_mrcwrite, artio_mrcread_header
%
% Utz H. Ermel 2018

    % Print action?
    if nargin == 1
        doPrint = 0;
    end
    
    % Read header
    [header, endian] = artia.mrc.read_header(fileName);
    xdim = header.nx;
    ydim = header.ny;
    zdim = header.nz;

    % Print action
    if doPrint
        fprintf('Reading MRC-file: %s with Dimensions:x=%g, y=%g, z=%g\n', fileName, xdim, ydim, zdim);
    end

    % Open file
    fid = fopen(fileName,'r',endian);

    % Skip header (512 bytes)
    fread(fid, 1024+header.next, 'int8');
    pixs = xdim*ydim*zdim;

    % Read data
    if header.mode == 0 % 8-bit signed integers
        data = fread(fid, pixs,'int8');
    elseif header.mode == 1 % 16-bit signed integers
        data = fread(fid, pixs,'int16');
    elseif header.mode == 2 % 32-bit float
        data = fread(fid, pixs,'float32');
    elseif header.mode == 6 % 16-bit unsigned integers
        data = fread(fid, pixs,'uint16');
    elseif ismember(header.mode, [3, 4])
        error('Mode is %d, but complex data types are not supported atm.', header.mode);
    elseif header.mode == 16
        error('Mode is %d, but RGB data is not supported atm.', header.mode);
    elseif header.mode == 101
        error('Mode is %d, but packed data is not supported atm.', header.mode);
    else
        error('Error: Unknown data type %d', header.mode);
    end
    
    % Reshape
    data = reshape(data, [xdim ydim zdim]);

    % Close file
    fclose(fid);
    
    % Done !
end