function tilts = dose_symmetric_tilts(countPerSide, increment, direction)
% artia.util.dose_symmetric_tilts returns the tilt angles of a tilt series
% acquired with the dose symmetric scheme.
%
% Parameters:
%   countPerSide (double):
%       Projections acquired in each tilting direction (not including the
%       0-degree projection.
%   increment (double):
%       Tilt increment in degrees.
%   direction (-1/1):
%       Tilt direction of the first tilt.
%
% Returns:
%   tilts (double[2*countPerSide + 1]):
%       The tilt angles in acquisition order.
%
% Author:
%   MK, 2018
%
    tilts = [];
    
    currentAng = 0;
    %direction = 1;
    
    for i = 1:countPerSide
        currentAng = -currentAng;
        tilts = [tilts currentAng];
        currentAng = currentAng + direction * increment;
        tilts = [tilts currentAng];
        direction = -direction;
    end
    currentAng = -currentAng;
    tilts = [tilts currentAng];
end