function ctfM = ctffind2emsart(inFile, outFile)
% artia.ctf.ctffind2emsart converts ctffind4 output to emsart CTF-files. 
% Writes an output em-file if output name is provided.
%
% Parameters:
%   inFile (str):
%       Path to the output of CTFFind. Should be run on the assembled,
%       sorted tilt stack that is also used for emsart, so the order is
%       correct. 
%   outFile (str):
%       Path to the output em-file.
%
% Returns:
%   ctfM (double[Mx5]):
%       CTF parameters for M projection in EmSART convention.
%
% Author:
%   UE, 2019
%

    fid = fopen(inFile, 'r');
    lines = {};
    i = 1;
    ctfM = zeros(1, 5);
    while ~feof(fid)
        line = fgetl(fid);
        if strcmp(line(1), '#')
            continue
        end
        
        vals = cell2mat(cellfun(@str2double, strsplit(line),'un',false));
        ctfM(i, 1) = vals(6);
        ctfM(i, 2) = vals(3)/10;
        ctfM(i, 3) = vals(2)/10;
        ctfM(i, 4) = (vals(2)/10)-(vals(3)/10);
        ctfM(i, 5) = deg2rad(vals(4));
        
        i = 1+1;
    end
    fclose(fid);
    
    if ~isempty(outFile)
        artia.em.write(ctfM, outFile);
    end
end

