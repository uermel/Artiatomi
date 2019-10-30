function mask = ellipsoid(dims, radius, sigma, center, steps, stepSize)
    
    if nargin < 5
        steps = dims(1)/2*10;
    end
    
    if nargin < 6
        stepSize = 0.1;
    end
    
    % Make dimensions vectors if necessary.
    if numel(dims) == 1
        dims = [dims dims dims];
    end
    
    if numel(center) == 1
        center = [center center center];
    end
    
    if numel(radius) == 1
        radius = [radius radius radius];
    end
    
    % Grid
    a = radius(1);
    b = radius(2);
    c = radius(3);
    bR = floor(dims/2);
    [x, y, z] = meshgrid(-bR(1):bR(1)-1, -bR(2):bR(2)-1, -bR(3):bR(3)-1);
    
    if sigma > 0
        % Gaussian for stepwise radius increase
        vx = 0:stepSize:steps*stepSize;
        v = exp(-(vx./sigma).^2);

        % Step up radius and assign pixels with appropriate value from
        % gaussian
        mask = zeros(dims);
        for j = 1:numel(v)
            i = vx(j);
            el = 1 >= x.^2/(a+i).^2 + y.^2/(b+i).^2 + z.^2/(c+i).^2;
            cond = mask == 0 & el;
            mask(cond) = v(j);
        end
        mask = mask./max(mask(:));

        % Gauss filter just to be sure
        [x, y, z] = meshgrid(-10:9, -10:9, -10:9);
        filt = exp(-(sqrt(x.^2 + y.^2 + z.^2)./1).^2);
        mask = convn(mask, filt, 'same');
        mask = mask./max(mask(:));

        % Remove extreme values
        mask(mask < exp(-4)) = 0;
    else
        mask = double(1 >= x.^2/(a).^2 + y.^2/(b).^2 + z.^2/(c).^2);
    end
    
    % Move to final position
    newCenter = center - (bR + 1);
    mask = move(mask, newCenter);
end