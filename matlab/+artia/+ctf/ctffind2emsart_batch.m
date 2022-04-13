function ctfM = ctffind2emsart_batch(inFile, outFile)
% artia.ctf.ctffind2emsart converts ctffind4 output to emsart CTF-files. 
% Writes an output em-file if output name is provided. 
%
% Suggested Workflow is to create the sorted tilt stack, then run ctffind
% on the stack and use this function to convert the output file.
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
%       CTF parameters for M projections in EmSART convention.
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
        
        defU = vals(3);
        defV = vals(2);
        defAng = vals(4);
        if defU < defV
            defl = defU/10;
            defh = defV/10; 
        else
            defl = defV/10;
            defh = defU/10;
            defAng = defAng + 90;
        end
        
        ctfM(i, 2) = defl;
        ctfM(i, 3) = defh;
        ctfM(i, 4) = (defh)-(defl);
        ctfM(i, 5) = deg2rad(defAng);
        
        i = 1+1;
    end
    fclose(fid);
    
    if ~isempty(outFile)
        artia.em.write(ctfM, outFile);
    end
end

