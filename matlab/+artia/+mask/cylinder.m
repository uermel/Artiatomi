function mask = cylinder(dims, radius, sigma, center, steps, stepSize)
    
    if nargin < 5
        steps = dims(1)/2*10;
    end
    
    if nargin < 6
        stepSize = 0.1;
    end
    
    if numel(dims) == 1
        dims = [dims dims dims];
    end
    
    if numel(center) == 1
        center = [center center center];
    end
    
    % Horizontal cylinder section
    xyC = floor(dims(1:2)/2) + 1;
    mask2 = artia.mask.ellipse(dims(1:2), radius(1:2), sigma, xyC);
    
    % Assemble cylinder and pad
    mask2 = reshape(mask2, [size(mask2) 1]);
    cyl = repmat(mask2, 1, 1, radius(3) * 2);
    dz = dims(3) - radius(3) * 2;
    before = dz/2 + 1;
    after = dz/2 - 1;
    mask = padarray(cyl, [0, 0, before], 0, 'pre');
    mask = padarray(mask, [0, 0, after], 0, 'post');
    
    % Gaussian at top and bottom
    if sigma > 0
        botInd = 1:before;
        topInd = before+radius(3)*2:before+radius(3)*2+after;
        
        botDist = fliplr(botInd);
        topDist = topInd - (before+radius(3)*2) + 1;
        
        botW = exp(-(botDist./sigma).^2);
        topW = exp(-(topDist./sigma).^2);
        
        for i = 1:numel(botInd)
            mask(:,:,botInd(i)) = botW(i) .* mask2;
        end
        
        for i = 1:numel(topInd)
            mask(:,:,topInd(i)) = topW(i) .* mask2;
        end   
    end
    
    % Move to final position
    bR = floor(dims./2);
    newCenter = center - (bR + 1);
    mask = move(mask, newCenter);
end