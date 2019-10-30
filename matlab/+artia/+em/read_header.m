function [header, endian, fid] = read_header(file)
% artia.em.read_header -- Read header of files in EM-format
%
% Usage:
%    [header, endian] = artia.em.read_header(fileName)
%
% Arguments:
%   fileName (str):           
%       File in EM-Format [character vector]
%
% Returns:
%    header (struct):
%        Matlab struct containing header info
%
%    endian (str):      
%       string indicating endianess
%
% See Also:
%   artia.em.read, artia.em.write
%
% Author:
%   Utz H. Ermel 2019
    

    % Is file or file ID?
    if ischar(file)
        fid = fopen(file,'r','ieee-le');
    else
        fid = file;
    end
    
    % Find out encoding
    mC = fread(fid, 1, 'int8');
    
    if mC == 1 || mC == 6
        endian = 'ieee-le';
        fseek(fid, 0, -1);
    else
        endian = 'ieee-be';
        fclose(fid);
        fid = fopen(fileName, 'r', endian);
    end
    
    % Start header
    header = struct();
    fmt = artia.em.header_fmt();
    el = fieldnames(fmt);
    for i = 1:numel(el)
        header.(el{i}) = fread(fid, fmt.(el{i}){2}, fmt.(el{i}){3});
    end
    
    % Is this a marker file?
    if header.isNewHeaderFormat == 0 % This is not a marker file
        header = rmfield(header, {'aliScore', ...
                                  'beamDeclination', ...
                                  'markerOffset', ...
                                  'magAnisoFactor', ...
                                  'magAnisoAngle', ...
                                  'imageSizeX', ...
                                  'imageSizeY'});
        header.isMarker = false;
    else
        header.isMarker = true;
    end
    
    % Close file if filename was supplied
    if ischar(file)
        fclose(fid);
        fid = [];
    end
    
    % Done !
end