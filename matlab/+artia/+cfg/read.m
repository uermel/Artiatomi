function outstruct = read(infile)
% artia.cfg.read creates MATLAB structs from Artiatomi cfg-files. All values 
% are stored as strings, including numerical entries.
%
% Usage:
%   outstruct = cfg2struct('path/to/file.cfg');
%
% Parameters:
%   infile (str):
%       Path to a config file.
%
% Returns:
%   outstruct (struct):
%       The corresponding structure.
%
% Author:
%   Utz H. Ermel, 2018    

    disp(['Reading cfg-file ' infile]);

    content = fileread(infile);
    pattern = '(\S*)\s*= *(.*)';
    params = regexp(content, pattern, 'tokens', 'dotexceptnewline');
    
    outstruct = struct();
    
    for i = 1:size(params, 2)
        if strcmp(params{i}{1}(1), '#')
            continue
        end
        outstruct.(params{i}{1}) = params{i}{2};
    end
end