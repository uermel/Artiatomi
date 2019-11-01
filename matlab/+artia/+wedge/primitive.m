function wedge = primitive(dims, minTilt, maxTilt)
% artia.wedge.primitive creates basic, binary missing wedge files given the
% minimum and maximum tilt angle of a tilt series. Tilt axis is y-axis.
%
% Parameters:
%   dims (double[3]):
%       Box size (of the particles)
%   minTilt (double):
%       minimum tilt angle of the series (degrees)
%   maxTilt (double):
%       maximum tilt angle of the series (degrees)
%
% Returns:
%   wedge (double[dims]):
%       The wedge file.
%
% Author:
%   ASF, 2006

%warning off MATLAB:divideByZero;
warning off;

switch(nargin)

    case 0
        error('Error: Not enough input arguments');

    case 1
        error('Error: Not enough input arguments');

    case 2

        minTilt = minTilt*pi/180;

        if size(dims,2)==2
            [x,y] = ndgrid(-floor(dims(1)/2):-floor(dims(1)/2)+dims(1)-1,...
                -floor(dims(2)/2):-floor(dims(2)/2)+dims(2)-1);
            wedge = ones(dims(1), dims(2));
            ind = find(tan(minTilt) > abs(x)./abs(y));
            wedge(ind)=0;

        elseif size(dims,2)==3
            [x,y,z] = ndgrid(-floor(dims(1)/2):-floor(dims(1)/2)+dims(1)-1,...
                -floor(dims(2)/2):-floor(dims(2)/2)+dims(2)-1,...
                -floor(dims(3)/2):-floor(dims(3)/2)+dims(3)-1);
            wedge = ones(dims(1), dims(2), dims(3));
            ind = find(tan(minTilt) > abs(x)./abs(z));
            wedge(ind)=0;

        else
            error('Error: Dimensions are not correct');
        end

    case 3
        if dims(1)~=dims(2) | dims(1)~=dims(3)
            error('Error: dimensions should be the same! (Cubic volume)')
        end

        if size(dims,2)==3
            %Create appropriate zero field
           
            maxangle = maxTilt*pi/180;
            minangle = minTilt*pi/180;
            [x,y,z] = ndgrid(-floor(dims(1)/2):-floor(dims(1)/2)+dims(1)-1,-floor(dims(2)/2):-floor(dims(2)/2)+dims(2)-1,-floor(dims(3)/2):-floor(dims(3)/2)+dims(3)-1);
            wedge = ones(dims(1), dims(2), dims(3));
            % 1st angles
            ind = find(tan(pi/2-maxangle) > (x)./(z));
            % merge with negative
            ind2=find(tan(-pi/2-minangle) < (x)./(z));
            ind = intersect(ind, ind2);
            wedge(ind)=0;
          

            
        end
end

warning on;




