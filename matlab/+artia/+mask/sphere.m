function mask=sphere(dims,radius,sigma,center)
% sphereGaussianBoundaries masks volume with sphere of radius r around center
%   vol=sphereGaussianBoundaries(vol, radius,sigma,center)
%
%INPUT
%   vol          : volume
%   radius       : radius determining radius of sphere
%   sigma       : smoothing of mask; if entered mask will be smoothened;
%                  every voxel outside radius gets smoothened by a gaussian
%                  function exp(-((r-radius)/simga)^2)
%   center       : vector determining center of sphere
%
%OUTPUT
%   vol          : masked volume
%
%EXAMPLE
%  
%   yyy = sphereGaussianBoundaries([64 64 64],4,10,[16 16 1]);
%   imagesc(yyy);

%
%08/14/02 FF
%last revision 
%25/03/04 FF
% JR

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

