function mask = ellipse(dims, radius, sigma, center, steps, stepSize)
% artia.mask.ellipse creates a elliptical density, optionally with an 
% approximately gaussian border. Density is positive. Gaussian is
% approximated by dilating the ellipse edge at small increments.
%
% Parameters:
%   dims (double[1]/double[2]):
%       The box dimensions. If one-dimensional, box is assumed cubic.
%   radius (double[1]/double[2]):
%       Radii of the 2 elliptical axes. If one-dimensional, ellipse is a
%       circle. 
%   sigma (double):
%       if ~= 0: every voxel outside radius gets smoothened by a gaussian
%                function exp(-((r-radius)/simga)^2)
%   steps (int):
%       Number of incremental steps at which to approximate the gaussian.
%   stepSize (double):
%       Size of the incremental steps.
%
% Returns:
%   mask (double):
%       The box containing the ellipsoid.
% 
% Author:
%   UE, 2019
%    
    if nargin < 5
        steps = dims(1)/2*10;
    end
    
    if nargin < 6
        stepSize = 0.1;
    end
    
    if numel(dims) == 1
        dims = [dims dims];
    end
    
    if numel(center) == 1
        center = [center center];
    end
    
    if numel(radius) == 1
        radius = [radius radius];
    end
    
    % Grid
    a = radius(1);
    b = radius(2);
    bR = floor(dims/2);
    [x, y] = meshgrid(-bR(1):bR(1)-1, -bR(2):bR(2)-1);
    
    if sigma > 0
        % Gaussian for stepwise radius increase
        vx = 0:stepSize:steps*stepSize;
        v = exp(-(vx./sigma).^2);

        % Step up radius and assign pixels with appropriate value from
        % gaussian
        mask = zeros(dims);
        for j = 1:numel(v)
            i = vx(j);
            el = 1 >= x.^2/(a+i).^2 + y.^2/(b+i).^2;
            cond = mask == 0 & el;
            mask(cond) = v(j);
        end
        mask = mask./max(mask(:));

        % Gauss filter just to be sure
        [x, y] = meshgrid(-10:9, -10:9);
        filt = exp(-(sqrt(x.^2 + y.^2)./1).^2);
        mask = convn(mask, filt, 'same');
        mask = mask./max(mask(:));

        % Remove extreme values
        mask(mask < exp(-4)) = 0;
    else
        mask = double(1 >= x.^2/(a).^2 + y.^2/(b).^2);
    end
    
    % Move to final position
    newCenter = center - (bR + 1);
    mask = move(mask, newCenter);
end