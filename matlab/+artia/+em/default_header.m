function header = default_header(varargin)
    
    % Set up default values for arguments
    header_fmt = artia.em.header_fmt();
    defs = struct();
    names = fieldnames(header_fmt);
    for i = 1:numel(names)
        defs.(names{i}).val = header_fmt.(names{i}){4};
    end
    
    artia.sys.getOpts(varargin, defs);
    
    % Read in values supplied to overwrite default values
    header = struct();
    for i = 1:numel(names)
        header.(names{i}) = eval(names{i});
    end
end