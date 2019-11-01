function omotl = text2motl(textFile, tomoNr, wedgeNr)
% artia.motl.text2motl converts a three column, tab-separated particle list
% to EmSART particle list format. 
%
% Parameters:
%   textFile (str):
%       Path to the input text file.
%   tomoNr:
%       Tomogram number of the tomogram the particles originate from.
%   wedgeNr:
%       Number of wedge volume to use during averaging.
%
% Returns:
%   omotl(double[20xN]):
%       The particle list.
%
% Author:
%   UE, 2019
%

    fid = fopen(textFile, 'r');
    C = textscan(fid, '%f\t%f\t%f');
    
    omotl = zeros(20, numel(C{1}));
    omotl(5, :) = tomoNr;
    omotl(6, :) = 1:numel(C{1});
    omotl(7, :) = tomoNr;
    omotl(8, :) = C{1};
    omotl(9, :) = C{2};
    omotl(10, :) = C{3};
    omotl(20, :) = 1;
    
    fclose(fid);
end

