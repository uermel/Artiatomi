function outstruct = read(infile)
% A function to convert MKTools config files to MATLAB structures. The
% structure contains a field for each entry in the config file.
%
% infile - Path to a config file
%
% outstruct - The corresponding structure.
%
% example: Reading a reconstruction config file.
%
% outstruct = cfg2struct('path/to/recon.cfg');
%
% outstruct = 
% 
%   struct with fields:
% 
%                CudaDeviceID: '1 2'
%              ProjectionFile: '/home/Group/millertree/yeast/cryo_pp/2018_01_04/tomos/tomo_12/tomo_12.st_Alig.st'
%               OutVolumeFile: '/home/Group/millertree/yeast/cryo_pp/2018_01_04/tomos/Reconstructions/tomo_12_2k_ss4.em'
%                  MarkerFile: '/home/Group/millertree/yeast/cryo_pp/2018_01_04/tomos/tomo_12/tomo_12_marker_post_alig.em'
%                               -
%                               -
%                               -
%           WriteVolumeAsFP16: 'false'
%       ProjectionScaleFactor: '1'
%     ProjectionNormalization: 'std'
%
% IMPORTANT: All values will be strings, including numerical entries.
%
% See also: struct2cfg, modify_cfgs
%
% UE 2018    
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