function data = read_inc(fileName, orig, width, varargin)
% artia.em.read_inc reads a portion of an EM-file.
%
% Parameters:
%   fileName (str):
%       Path to the input file.
%   orig (double[1] or double[3]):
%       1. Single value: is interpreted as the index of the z-section to be
%       read.
%       2. 3-element vector: is interpreted as the origin of the box to be
%       read (see Name Value pair 'readMode').
%   width (double[3]):
%       The dimensions of the subvolume to be read.
%
% Name Value Pairs:
%   'readMode' (str):
%       1. 'CenterWidth' (default): The orig-parameter specifies the
%       center of the box to be cut in EmSART convention.
%       2. 'CornerWidth': The orig-parameter specifies the upper left
%       corner of the box to be cut.
%   'padMode' (str):
%       If part of the box is outside the volume:
%       1. 'replicate' (default): the void is filled by replicating the border voxels.
%       2. 'symmetric': the void is filled by mirroring the filled portion.
%       3. 'circular': the void is filled with a circular repetition of the filled portion
%       4. 'mean': the void is filled with the mean value of the filled
%       portion.
%       5. 'zero': the void is filled with zeros.
%
% Returns:
%   data (double):
%       The requested subvolume.
%
% Author:
%   UE, 2019
%
    % Default params
    defs = struct();
    defs.readMode.val = 'CenterWidth';
    defs.padMode.val = 'replicate';
    artia.sys.getOpts(varargin, defs);
    
    % Check if sections should be read, otherwise check if input has
    % correct dimensions
    if numel(orig) == 1
        assert(isempty(width), 'Width cannot be specified if reading z-sections.')
        readMode = 'section';
    else
        assert(numel(orig) == 3, 'Origin has to be 3-element vector. Is %d-element vector.', numel(orig));
        assert(numel(width) == 3, 'Width has to be 3-element vector. Is %d-element vector.', numel(width));
        orig = reshape(orig, 3, 1);
        width = reshape(width, 3, 1);
    end

    % Read header
    [header, ~] = artia.em.read_header(fileName);
    dim = zeros(3, 1);
    dim(1) = header.dimX;
    dim(2) = header.dimY;
    dim(3) = header.dimZ;
    
    % Compute box coordinates
    switch lower(readMode)
        case 'centerwidth'
            r = width/2;
            tmi = orig - r;
            tma = orig + (r-1);
            
        case 'cornerwidth'
            tmi = orig;
            tma = orig + width - 1;
            
        case 'section'
            tmi = zeros(3, 1);
            tmi(3) = orig;
            tma = [dim(1) dim(2) 1];
    end
    
    % Check that at least 1 pixel is inside the tomogram
    if any(tmi > dim)
        error('Upper left corner at [%d %d %d]. Cannot exceed [%d %d %d].', ...
              tmi(1), tmi(2), tmi(3), dim(1), dim(2), dim(3));
    end
    
    if any(tma < 1)
        error('Lower right corner at [%d %d %d]. Cannot be below [1 1 1].', ...
              tma(1), tma(2), tma(3));
    end
    
    % Get memmap
    mmap = artia.em.memmap(fileName);
    
    % Init Padding
    pmi = zeros(3, 1);
    pma = zeros(3, 1);

    % Compute padding
    for j = 1:3
        if tmi(j) < 1
            pmi(j) = abs(tmi(j)) + 1;
            tmi(j) = 1;
        end

        if tma(j) > dim(j)
            pma(j) = tma(j) - dim(j);
            tma(j) = dim(j);
        end
    end
    
    % Read
    data = mmap.Data.data(tmi(1):tma(1), ...
                          tmi(2):tma(2), ...
                          tmi(3):tma(3));

    % Padval
    switch lower(padMode)
        case 'replicate'
            padval = 'replicate';
        case 'symmetric'
            padval = 'symmetric';
        case 'circular'
            padval = 'circular';
        case 'mean'
            padval = mean(data(:));
        case 'zero'
            padval = 0;
    end

    % Pad
    if any(pmi > 0)
        data = padarray(data, pmi, padval, 'pre');
    end
    if any(pma > 0)
        data = padarray(data, pma, padval, 'post');
    end
    
    % Done!
end