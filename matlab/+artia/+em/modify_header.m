function modify_header(file, varargin)

    % Set up default values for arguments
    fmt = artia.em.header_fmt();
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
    if ischar(file)
        fid = fopen(file, 'a+', 'ieee-le');
    else
        fid = file;
    end
    
    for i = 1:numel(sec_names)
        fseek(fid, fmt.(sec_names{i}){1}, 'bof');
        fwrite(fid, mod.(sec_names{i}), fmt.(sec_names{i}){3});
    end
    
    % Close if filename was passed
    if ischar(file)
        fclose(fid);
    end
    
    % Done!
end