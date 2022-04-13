function vals = ext2arr(extended)
% artia.mrc.ext2arr converts the struct containing extended header info to
% a matrix that can be saved to file. This should only be needed inside the
% actual header writing functions, never elsewhere.
%
% Parameters:
%   extended (struct):
%       The extended header field from a MRC-header struct.
%
% Returns:
%   vals (double[32x1024]):
%       The extended header data in matrix form.
%
% Author:
%   UE, 2019
%
    % FEI/Kunz header
    names = {'tiltAngle', ...
            'tiltAngleBeta', ...
            'stagePos', ...
            'imageShift', ...
            'defocus', ...
            'expTime', ...
            'meanInt', ...
            'tiltAxis', ...
            'pixelSize', ...
            'magnification', ...
            'remainder'};
    vars = [1 1 3 2 1 1 1 1 1 1 19];   
    starts = vars(1);
    ends = vars(1);
    for i = 2:numel(vars)
        starts(i) = ends(i-1) + 1;
        ends(i) =  starts(i) + vars(i) - 1;
    end
    
    % Init array
    vals = zeros(sum(vars), 1024);
    
    % Write vals to array
    for i = 1:numel(names)
        vals(starts(i):ends(i), :) = extended.(names{i});
    end
end