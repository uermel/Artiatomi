function mrcfid = write(data, fileName, angpix, varargin)
% artia.mrc.write writes the 1-3 dimensional array to an MRC-file at fileName. 
%
% Usage:
%       
%   artia.mrc.write(data, fileName, 'doPrint', false, 'dataType', 
%   'float32', 'header', [], 'doAppend', 'false');
%
% Arguments:
%   data (int, double):
%       1-, 2- or 3-dimensional array of data to be written
%
%   fileName (str):
%       Name of output file.
%
% Name Value Pairs:
%   doPrint (bool):
%       False to suppress output of filename/dimensions during saving.
%       Default: false
%   
%   dataType (str):
%       Any type code among {'int8', 'int16', 'float32', 'uint16'}. Default: 'float32'
%
%   header (struct):
%       Matlab structure defining header. See :func:`+artia.+mrc.header_fmt`
%       for details. Default: []
%
%   tiltAngles (double[N]):
%       Array containing tilt angles where N is the number of projections
%       (zdim).
%
%   doAppend (bool):
%       If true, data will be appended at the end of the specified file.
%       Dimensions in the header are corrected. If false, a new header is
%       written first, overwriting any existing content. Default: false
%
%   returnFid (bool):
%       If true, mrc-file isn't closed after writing header and data and
%       matlab file-ID is returned.
%
% Author:
%   UE, 2019

    % Default params
    defs = struct();
    defs.doPrint.val = false;
    defs.dataType.val = 'float32';
    defs.header.val = [];
    defs.tiltAngles.val = [];
    defs.doAppend.val = false;
    defs.returnFid.val = false;
    artia.sys.getOpts(varargin, defs);
    
    % Data size
    [xdim, ydim, zdim] = size(data);
    
    % Figure out data type    
    switch dataType
        case 'int8'
            dataTypeCode = 0;
        case 'int16'
            dataTypeCode = 1;
        case 'float32'
            dataTypeCode = 2;
        case 'uint16'
            dataTypeCode = 6;
    end
    
    % If appending, read header now do sanity check
    if doAppend 
        old_header = artia.mrc.read_header(fileName);
        if ~strcmp(old_header.mode, dataTypeCode)
            error('Data of type %s cannot be appended to file of type %s', dataType, old_header.mode);
        end
        
        if old_header.nx ~= xdim && old_header.ny ~= ydim
            error(['Dimension mismatch. Data to be appended has dimensions [%d, %d]' ...
                  'while file has dimensions [%d, %d]. Must match.'], xdim, ydim, ...
                  old_header.nx, old_header.ny);
        end
        
        new_dim = old_header.nz + zdim;
    end
    
    % If no further items provided simply generate default header with data
    % type
    if isempty(header)
        header = artia.mrc.default_header('nx', xdim, ...
                                          'ny', ydim, ...
                                          'nz', zdim, ...
                                          'mx', xdim, ...
                                          'my', ydim, ...
                                          'mz', zdim, ...
                                          'xlen', xdim * angpix, ...
                                          'ylen', ydim * angpix, ...
                                          'zlen', zdim * angpix, ...
                                          'mode', dataTypeCode);
    end
    
    if ~isempty(tiltAngles)
        tilts = zeros(1, 1024);
        tilts(1:numel(tiltAngles)) = tiltAngles;
        header.extended = artia.mrc.fei_extended('tiltAngle', tilts);
    end
    
    % Print action
    if doPrint
        fprintf('Writing MRC-file: %s\n', fileName);
    end
    
    % Write header if not appending
    if doAppend 
        fid = fopen(fileName, 'a+', 'ieee-le');
        fwrite(fid, data, dataType);
        artia.mrc.modify_header(fid, 'nz', new_dim, 'mz', new_dim, 'zlen', new_dim * angpix);
    else
        fid = fopen(fileName, 'w', 'ieee-le');
        fid = artia.mrc.write_header(header, fid);
        fwrite(fid, data, dataType);
    end
    
    % Close file if not returning fid
    if returnFid
        mrcfid = fid;
    else
        fclose(fid);
        mrcfid = [];
    end
end