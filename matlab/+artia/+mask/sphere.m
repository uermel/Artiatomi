function mask=sphere(dims,radius,sigma,center)
% artia.mask.sphere creates a spherical density, optionally with a gaussian
% border. Density is positive.
%
% Parameters:
%   dims (double[3]): 
%       The box dimensions.
%   radius (double):
%       The radius of the sphere.
%   sigma (double):
%       if ~= 0: every voxel outside radius gets smoothened by a gaussian
%                function exp(-((r-radius)/simga)^2)
%   center (double[3]):
%       The center of the sphere inside the box.
%
% Returns:
%   mask (double[dims]):
%       The box containing the sphere.
%
% Author:
%   FF, 2004; JR, 2019
%

narginchk(1,4)
%error(nargchk(1,4,nargin))
if (nargin < 4)
    center=[floor(dims(1)/2)+1, floor(dims(2)/2)+1, floor(dims(3)/2)+1];
end
[x,y,z]=ndgrid(0:dims(1)-1,0:dims(2)-1,0:dims(3)-1);
if (nargin < 2)
    radius = floor((min(min(dims(1),dims(2)),dims(3))-1)/2) ;
end
x = sqrt((x+1-center(1)).^2+(y+1-center(2)).^2+(z+1-center(3)).^2);
ind = find(x>radius);
clear y z;
mask = ones(dims(1), dims(2), dims(3));

mask(ind) = 0;
if (nargin > 2) 
    if (sigma > 0)
        mask(ind) = exp(-((x(ind) -radius)/sigma).^2);
        ind = find(mask < exp(-4));
        mask(ind) = 0;
    end
end

