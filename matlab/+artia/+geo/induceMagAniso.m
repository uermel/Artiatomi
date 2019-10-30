function distorted = induceMagAniso(coords, amount, angle, dims)
% artia.geo.induceMagAniso applies an affine transformation to induce
% magnification anisotropy.
%
%
    % Get distortion matrices -- these are the matrices for correcting
    % distortion, so inversion of the stretch is necessary later on
    [Sh2, Ro2, St, Ro1, Sh1] = distortionMatrices(angle, amount, dims(1), dims(2));
    
    % Transpose if necessary
    if size(coords, 2) ~= 2
        coords = coords';
    end
    
    % Init output
    distorted = zeros(size(coords));
    
    % Invert the stretch to induce distortion
    St(1, 1) = 1/St(1,1);
    M = Sh2 * Ro2 * St * Ro1 * Sh1;
    
    % Apply the distortion
    for i = 1:size(coords, 1)
        vec = [coords(i, 1), coords(i, 2), 1]';
        
        res = M * vec;
        
        distorted(i, 1) = res(1);
        distorted(i, 2) = res(2);
    end
end