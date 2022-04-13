function omotl = tbl2motl(textFile, tomoNr, wedgeNr)
% artia.motl.tbl2motl converts a three column particle list of some table
% format taken by the Matlab readtable() function, such as .txt or .csv,
% into EmSART particle list format. This is a more generic version of the
% artia.motl.text2motl function.
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
%   KS, 2020
%
    C = table2array(readtable(textFile, 'ReadVariableNames',false));
    
    % readtable may not be able to detect the correct delimiter, especially
    % when there is only one row of data. In such a case, try the three
    % most commmon cases
    if numel(C(1,:)) < 3
        C = table2array(readtable(textFile, 'Delimiter', ' ', ...
            'ReadVariableNames',false));
    end
    if numel(C(1,:)) < 3
        C = table2array(readtable(textFile, 'Delimiter', '\t', ...
            'ReadVariableNames',false));
    end
    if numel(C(1,:)) < 3
        C = table2array(readtable(textFile, 'Delimiter', ',', ...
            'ReadVariableNames',false));
    end
    
    omotl = zeros(20, numel(C(:,1)));
    omotl(5, :) = tomoNr;
    omotl(6, :) = 1:numel(C(:,1));
    omotl(7, :) = tomoNr;
    omotl(8, :) = C(:,1);
    omotl(9, :) = C(:,2);
    omotl(10, :) = C(:,3);
    omotl(20, :) = 1;
    
end

