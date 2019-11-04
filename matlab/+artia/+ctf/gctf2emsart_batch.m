function ctfM = gctf2emsart_batch(starFile, outFile, varargin)
% artia.ctf.gctf2emsart converts the output STAR-file of GCTF to emsart convention.
% This function assumes, that GCTF is run in batch on the individual
% projection images, yielding one STAR-file containing all projections. 
% The input values can be sorted using different methods.
%
% Suggested workflow is to store all projections in one directory, run GCTF in
% batch on all those file and use the STAR-output option. Then convert the
% STAR-file using this function.
%
% Parameters:
%   starFile (str):
%       Path to the input STAR file.
%   outFile (str):
%       Path to the output EmSART CTF-file (EM-format).
%
% Name Value Pairs:
%   'sortMode' (['regex', 'idx', 'skip']):
%       Specifies how the entries of the STAR file are sorted prior to
%       storing in the EmSART-ctf file.
%       1. 'regex' - User provided regular expression pattern is used to
%       derive the tilt angle from each file name, data is then sorted by
%       tilt angle in ascending order. Provided as Name Value pair
%       'regexPattern'.
%       2. 'idx' - User provided index is used to sort the data. Provided
%       as Name Value pair 'order'.
%       3. 'skip' - Data is assumed to appear in the same order in the STAR
%       file as in the tilt stack and no sorting is performed.
%   'regexPattern' (str):
%       Pattern to use to derive tilt angle from Micrograph names in the
%       star file if using sortMode 'regex'.
%   'order' (str):
%       Index to sort data by if using sortMode 'idx'.
%
% Returns:
%   ctfM (double[Mx5]):
%       CTF parameters for M projections in EmSART convention.
%
% Author:
%   UE, 2019

    % Default params
    defs = struct();
    defs.sortMode.val = 'regex';
    defs.regexPattern.val = '';
    defs.order.val = [];
    artia.sys.getOpts(varargin, defs)
    
    % Get data
    fields = {'MicrographName', 'DefocusU', 'DefocusV', 'DefocusAngle', 'CtfFigureOfMerit'};
    fmt = {'str', 'float', 'float', 'float'};
    [query, ~] = artia.star.read(starFile, fields, fmt);
    
    % Figure out order
    switch sortMode
        case 'regex'
            assert(~isempty(regexPattern), 'Must provide a regex pattern if using sortMode ''regex''.');
            angs = [];
            for i = 1:numel(query.MicrographName)
                tok = regexp(query.MicrographName{i}, regexPattern, 'tokens');
                angs = [angs str2double(tok{1}{1})];
            end
            [~, order] = sort(angs, 'ascend');
        case 'skip'
            order = 1:numel(query.DefocusU);
        case 'idx'
            order = order;
    end
    
    % Convert
    ctfM = zeros(numel(query.DefocusU), 5);
    
    defcc = query.CtfFigureOfMerit(order);
    defu = query.DefocusU(order)/10;
    defv = query.DefocusV(order)/10;
    defa = query.DefocusAngle(order);
    
    defl = zeros(numel(defu), 1);
    defh = zeros(numel(defu), 1);
    
    for i = 1:numel(defu)
        if defu(i) < defv(i)
            defl(i) = defu(i);
            defh(i) = defv(i); 
        else
            defl(i) = defv(i);
            defh(i) = defu(i);
            defa(i) = defa(i) + 90;
        end
    end
    
    astm = defh-defl;
    ctfM(:, 1) = defcc;
    ctfM(:, 2) = defl;
    ctfM(:, 3) = defh;
    ctfM(:, 4) = astm;
    ctfM(:, 5) = deg2rad(defa);
    
    if ~isempty(outFile)
        artia.em.write(ctfM, outFile);
    end
end

