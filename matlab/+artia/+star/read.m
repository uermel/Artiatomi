function [query_struct, result_struct] = read(file, arglist, formatlist)
% artia.star.read parses single-block star files and returns their contents
% in a (optionally) formatted matlab structure.
%
% Usage:
% [query_struct, result_struct] = e2r_star2struct(file, arglist, formatlist)
%
% Parameters:
%   file (str):          
%       Path to the STAR-file
%
%   arglist (cell):
%       Cell array of string variable names to be converted to
%       data types specified in formatlist and returned in
%       query_struct
%   formatlist (cell):
%       Cell array of strings. For each position in arglist,
%       one of 'str', 'float' or 'int' has to be provided. Read
%       data of the variables in arglist will be converted
%       accordingly.
%
% Returns:
%   query_struct (struct):
%       Struct with fields specified by arglist and data types
%       specified by formatlist
%
%   result_struct (struct):
%       Struct containing all fields present in file and all
%       values in string format
%
% Author:
%   UE, 2019

    fid = fopen(file, 'rt');
    
    %%% Find loop statement 
    counter = 0;
    loop = false;
    while ~loop
        line = fgetl(fid);
        counter = counter + 1;
        if contains(line, 'loop')
            loop = true;
        end
    end
    
    %%% Read fields
    us = true;
    vars = {};
    expr = '_rln([\w]*)';
    while us
        line = fgetl(fid);
        counter = counter + 1;
        if line(1) ~= '_'
            us = false;
        else
            match = regexp(line, expr, 'tokens');
            vars{end+1} = match{1}{1};
        end
    end
    
    %%% Rewind to BOF, then use textscan and generated format string to
    %%% read all values
    format = ['%s' repmat('\t%s', 1, numel(vars)-1)];
    frewind(fid);
    contents = textscan(fid, format, 'headerlines', counter-1);
    
    %%% Generate output stucts (query_struct contains user formatted data)
    result_struct = struct();
    query_struct = struct();
    for i = 1:numel(vars)
        result_struct.(vars{i}) = contents{i};
        
        is_arg = strcmp(arglist, vars{i});
        if any(is_arg)
            idx = find(is_arg);
            if strcmp(formatlist(idx), 'str')
                query_struct.(vars{i}) = contents{i};
            elseif strcmp(formatlist(idx), 'float')
                query_struct.(vars{i}) = str2double(contents{i});
            elseif strcmp(formatlist(idx), 'int')
                query_struct.(vars{i}) = int32(str2double(contents{i}));
            end
        end
    end
    fclose(fid);
end