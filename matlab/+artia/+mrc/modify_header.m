function modify_header(fileName, varargin)
% artia.mrc.modify_header modifies the header of an MRC-file
%
% Parameters:
%   file (str):
%       Path to the file.
%
% Name Value Pairs:
%   Header fields to be modified are supplied as Name Value pairs.
%
% Author:
%   UE, 2019
%

    % Set up default values for arguments
    fmt = artia.mrc.header_fmt();
    defs = struct();
    names = fieldnames(fmt);
    for i = 1:numel(names)
        defs.(names{i}).val = fmt.(names{i}){4};
    end
    
    [~, ud] = artia.sys.getOpts(varargin, defs);
    
    % Read in values supplied to modify
    mod = struct();
    for i = 1:numel(names)
        if ~ismember(names{i}, ud)
            mod.(names{i}) = eval(names{i});
        end
    end
    
    sec_names = fieldnames(mod);
    
    % Assert name/number of elements correct
    for i = 1 : numel(sec_names)        
        if numel(mod.(sec_names{i})) > fmt.(sec_names{i}){2} || numel(mod.(sec_names{i})) < fmt.(sec_names{i}){2}
            error('Wrong number of elements (%d) provided for section %s. Expected %d.', ...
                  numel(mod.(sec_names{i})), ...
                  sec_names{i}, ...
                  fmt.(sec_names{i}){2});
        end
    end
    
    % Open file if filename was passed
    if ischar(fileName)
        fid = fopen(fileName, 'a+', 'ieee-le');
    else
        fid = fileName;
    end
    
    for i = 1:numel(sec_names)
        fseek(fid, fmt.(sec_names{i}){1}, 'bof');
        fwrite(fid, mod.(sec_names{i}), fmt.(sec_names{i}){3});
    end
    
    % Close if filename was passed
    if ischar(fileName)
        fclose(fid);
    end
    
    % Done!
end