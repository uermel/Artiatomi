function ext = fei_extended(varargin)
% artia.mrc.fei_extended sets up an empty extended header struct in FEI
% convention. Values can be overwritten using Name Value pairs. All
% provided arrays have to be of size Nx1024.
%
% Name Value Pairs:
%   tiltAngle
%   tiltAngleBeta
%   stagePos
%   imageShift
%   defocus
%   expTime
%   meanInt
%   tiltAxis
%   pixelSize
%   magnification
%   remainder
%
% Returns:
%   ext (struct):
%       Struct containing the extended header fields. If not overwritten,
%       they're initialized to 0.
%
% Author:
%   UE, 2019

    % Set up default values for arguments
    defs = struct();
    defs.tiltAngle.val = zeros(1, 1024);
    defs.tiltAngleBeta.val = zeros(1, 1024);
    defs.stagePos.val = zeros(3, 1024);
    defs.imageShift.val = zeros(2, 1024);
    defs.defocus.val = zeros(1, 1024);
    defs.expTime.val = zeros(1, 1024);
    defs.meanInt.val = zeros(1, 1024);
    defs.tiltAxis.val = zeros(1, 1024);
    defs.pixelSize.val = zeros(1, 1024);
    defs.magnification.val = zeros(1, 1024);
    defs.remainder.val = zeros(19, 1024);
    
    artia.sys.getOpts(varargin, defs);
    
    % Read in values supplied to overwrite default values
    ext = struct();
    names = fieldnames(defs);
    for i = 1:numel(names)
        ext.(names{i}) = eval(names{i});
    end
end

