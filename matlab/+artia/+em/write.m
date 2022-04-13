
function emfid = write(data, fileName, varargin)
% artia.em.write writes the 1-3 dimensional array to an EM-file at fileName. 
%
% Usage:
%       
%   artia.em.write(data, fileName, 'doPrint', false, 'dataType', 
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
%       Any type code among {'int8', 'int16', 'int32', 'float32',
%       'float64'}. Default: 'float32'
%
%   header (struct):
%       Matlab structure defining header. See :func:`+artia.+em.header_fmt`
%       for details. Default: []
%
%   doAppend (bool):
%       If true, data will be appended at the end of the specified file.
%       Dimensions in the header are corrected. If false, a new header is
%       written first, overwriting any existing content. Default: false
%
%   returnFid (bool):
%       If true, em-file isn't closed after writing header and data and
%       matlab file-ID is returned.
%
% Author:
%   UE, 2019

    % Default params
    defs = struct();
    defs.doPrint.val = false;
    defs.dataType.val = 'float32';
    defs.header.val = [];
    defs.doAppend.val = false;
    defs.returnFid.val = false;
    artia.sys.getOpts(varargin, defs);
    
    % Data size
    [xdim, ydim, zdim] = size(data);
    
    % Figure out data type    
    switch dataType
        case 'int8'
            dataTypeCode = 1;
        case 'int16'
            dataTypeCode = 2;
        case 'int32'
            dataTypeCode = 4;
        case 'float32'
            dataTypeCode = 5;
        case 'float64'
            dataTypeCode = 9;
    end
    
    % If appending, read header now to sanity check
    if doAppend 
        [old_header] = artia.em.read_header(fileName);
        if ~strcmp(old_header.dataType, dataTypeCode)
            error('Data of type %s cannot be appended to file of type %s', dataType, old_header.dataType);
        end
        
        if old_header.dimX ~= xdim && old_header.dimY ~= ydim
            error(['Dimension mismatch. Data to be appended has dimensions [%d, %d]' ...
                  'while file has dimensions [%d, %d]. Must match.'], xdim, ydim, ...
                  old_header.dimX, old_header.dimY);
        end
        
        new_dim = old_header.dimZ + zdim;
    end
    
    % If no further items provided simply generate default header with data
    % type
    if isempty(header)
        header = artia.em.default_header('dataType', dataTypeCode);
    end
        
    % Print action
    if doPrint
        fprintf('Writing EM-file: %s\n', fileName);
    end
    
    % Prepare
    header.dimX = xdim;
    header.dimY = ydim;
    header.dimZ = zdim;
    
    % Write header if not appending
    if doAppend 
        fid = fopen(fileName, 'a+', 'ieee-le');
        fwrite(fid, data, dataType);
        artia.em.modify_header(fid, 'dimZ', new_dim);
    else
        fid = fopen(fileName, 'w', 'ieee-le');
        fid = artia.em.write_header(header, fid);
        fwrite(fid, data, dataType);
    end
    
    % Close file if not returning fid
    if returnFid
        emfid = fid;
    else
        fclose(fid);
        emfid = [];
    end
    
end