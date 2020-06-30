function fid = write_header(header, file)
% artia.em.write_header writes the header of an em-file.
%
% Parameters:
%   header (struct):
%       Matlab struct containing the em-header sections as fields.
%
%   fileName (str/fileID):
%       1. String input: Path to the file.
%       2. File ID input: Matlab file ID to an opened file.
%
% Returns:
%   fid (fileID):
%       Matlab file ID to the file.
%
% Author:
%    UE, 2019
%
    if ischar(file)
        fid = fopen(file,'w','ieee-le');
    else
        fid = file;
    end
    
    %fid = fopen(file,'w','ieee-le');
    
    header_fmt = artia.em.header_fmt();
    names = fieldnames(header_fmt);
    
    for i = 1:numel(names)
        fwrite(fid, header.(names{i}), header_fmt.(names{i}){3});
    end
    
    
     % Close file if filename was supplied
    if ischar(file)
        fclose(fid);
        fid = [];
    end
    %fclose(fid);
end