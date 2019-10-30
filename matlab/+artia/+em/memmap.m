function mmap = memmap(fileName, varargin)

    % Default params
    defs = struct();
    defs.mode.val = 'r';
    artia.sys.getOpts(varargin, defs);
    
    switch lower(mode)
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