function mmap = memmap(fileName, varargin)
% artia.em.mmap opens an EM-file as a memory map and returns the
% memorymap-object.
%
% Parameters:
%   fileName (str):
%       Path to the EM-file.
%
% Name Value Pairs:
%   'fmode' (str):
%       The file mode. Can be either 'r' (read) or 'w' (write). Defaults to
%       'r'.
%
% Returns:
%   mmap (object):
%       Matlab memory map object.
%
% Author
%   UE, 2019
%
    % Default params
    defs = struct();
    defs.fmode.val = 'r';
    artia.sys.getOpts(varargin, defs);
    
    switch lower(fmode)
        case 'r'
            filemode = false;
        case 'w'
            filemode = true;
    end

    % Open file
    fid = fopen(fileName,'r','ieee-le');

    % Read header
    [header, ~, fid] = artia.em.read_header(fid);
    xdim = header.dimX;
    ydim = header.dimY;
    zdim = header.dimZ;
    
    switch header.dataType
        case 1
            dataType = 'int8';
        case 2
            dataType = 'int16';
        case 4
            dataType = 'int32';
        case 5
            dataType = 'single';
        case 9
            dataType = 'double';
    end

    % Close file
    fclose(fid);
    
    % Open memory map
    mmap = memmapfile(fileName, 'Writable', filemode, ...
                                'Offset', 512, ...
                                'Format', {dataType, [xdim, ydim, zdim], 'data'});
end