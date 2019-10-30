function [results, usingDefaults] = getOpts(input, defaults)
% artia.sys.getOpts creates an inputParser object and parses the cell
% array input based on default values stored in the struct defaults. After
% parsing variables are assigned to the function workspace from which
% getOpts was called.
%
% Usage:
%   artia.sys.getOpts(input, defaults)
%
% Arguments:
%   input (cell):
%       The input items to be parsed (typically 'varargin' from 
%       the caller function).
%
%   defaults (struct):
%       Matlab structure containing the parameter names and
%       their default values. Should be of the following shape:
%       
%       .. code-block:: matlab   
%
%           defaults.('paramName').val = defaultValue;
%
% Example:   
%   Let add(a, b, c) be a function with required parameter a and
%   optional parameters b, c. getOpts can be used as follows:
%
%   .. code-block:: matlab
%
%       function add(a, varargin)
%           defs = struct();
%           defs.b.val = 5;
%           defs.c.val = 6;
%           artia.sys.getOpts(varargin, defs);
%           ...
%        end
%
% Author:
%   Utz H. Ermel 2019

    parser = inputParser;
    names = fieldnames(defaults);
    for i = 1:numel(names)
        parser.addParameter(names{i}, defaults.(names{i}).val);
    end
    
    parser.parse(input{:});
    
    for i = 1:numel(names)
        assignin('caller', names{i}, parser.Results.(names{i}))
    end
    
    results = parser.Results;
    usingDefaults = parser.UsingDefaults;
end