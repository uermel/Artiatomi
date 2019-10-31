function header = default_header(varargin)
% artia.em.header returns the default em-header as a matlab struct and
% replaces the default values with supplied name value pair parameters.
%
% Usage:
%   For a default header call
%       .. code-block:: matlab   
%       
%           header = default_header();
%
%   To override defaults call
%       .. code-block:: matlab
%
%           header = default_header('HeaderSection1', value, ...
%                                   'HeaderSection2', value)
%
%
% Name Value Pairs:
%   All header values to override are provided as Name-Value-Pairs.
%
% Returns:
%   header (struct):
%       Matlab structure containing the header sections of an EM-file as
%       fields. Fields are in order of the appearance in the header.
%
% Author:
%   UE, 2019
%
    % Set up default values for arguments
    header_fmt = artia.em.header_fmt();
    defs = struct();
    names = fieldnames(header_fmt);
    for i = 1:numel(names)
        defs.(names{i}).val = header_fmt.(names{i}){4};
    end
    
    artia.sys.getOpts(varargin, defs);
    
    % Read in values supplied to overwrite default values
    header = struct();
    for i = 1:numel(names)
        header.(names{i}) = eval(names{i});
    end
end