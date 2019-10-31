function write(instruct, outfile)
% artia.cfg.write writes the 
% A function to convert MATLAB structures to MKTools config files. The
% structure must contain fields with string values only. The entry names
% will be the field names
%
% instruct - A structure containing fields with names of entries and string
%            values.
% example: 
% intstruct = 
% 
%   struct with fields:
% 
%                CudaDeviceID: '1 2'
%              ProjectionFile: '/home/Group/millertree/yeast/cryo_pp/2018_01_04/tomos/tomo_12/tomo_12.st_Alig.st'
%               OutVolumeFile: '/home/Group/millertree/yeast/cryo_pp/2018_01_04/tomos/Reconstructions/tomo_12_2k_ss4.em'
%                  MarkerFile: '/home/Group/millertree/yeast/cryo_pp/2018_01_04/tomos/tomo_12/tomo_12_marker_post_alig.em'
%
% outfile - Path to a config file to create
%
% example: Writing a reconstruction config file.
%
% struct2cfg(instruct, 'path/to/recon.cfg');
%
% IMPORTANT: All values must be strings, including numerical entries. 
% IMPORTANT: Entry names are not checked for spelling.
%
% See also: cfg2struct, modify_cfgs
%
% UE 2018

    fid = fopen(outfile,'wt');
    params = fieldnames(instruct);
    
    
    for i = 1:numel(params)
        fprintf(fid, '%s = %s\n', params{i}, instruct.(params{i}));
    end
    fclose(fid);
    
    disp(['Wrote cfg-file ' outfile]);
end