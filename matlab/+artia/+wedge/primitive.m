function wedge = primitive(dims, minTilt, maxTilt)
%   wedge=createWedge(dims,angle,angle2)
%
%   WEDGE produces a wedge shaped array.
%   This array can be used as a window filter in Fourier space...
%
%   dims    Dimensions of the image where the wedge has to be created
%   angle   semi angle of missing wedge in deg (minimum angle +64)
%   angle2  In case of an aymmetric wedge (maximum  angle e.g. -64)
%   wedge   output - filter
%   Direction of the wedge towards y axis

%Orientation has not been tested: Use eventually: mirror(w,'z')

%ASF 07 01 06 Asymmetric wedges

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




