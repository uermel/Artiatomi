function write(instruct, outfile)
% artia.cfg.write generates Artiatomi cfg-files given a struct with input
% values. All values need to be strings.
%
% Usage:
%   artia.cfg.write(instruct, outfile)
%
% Parameters:
%   instruct (struct):
%       A structure containing fields with names of entries and string
%       values.
% Author:
%   Utz H. Ermel 2018

    fid = fopen(outfile,'wt');
    params = fieldnames(instruct);
    
    
    for i = 1:numel(params)
        fprintf(fid, '%s = %s\n', params{i}, instruct.(params{i}));
    end
    fclose(fid);
    
    disp(['Wrote cfg-file ' outfile]);
end