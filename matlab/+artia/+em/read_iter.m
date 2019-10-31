function iter = read_iter(fileName, orig, width, varargin)
% artia.em.read_iter returns a function handle to 

    % Default params
    defs = struct();
    defs.readMode.val = 'CenterWidth';
    defs.padMode.val = 'replicate';
    artia.sys.getOpts(varargin, defs)
    
    % Check if sections should be read, otherwise check if input has
    % correct dimensions
    if numel(size(orig)) == 2
        assert(any(size(orig) == 1), 'Origin has to be vector if reading z-sections.')
        assert(isempty(width), 'Width cannot be specified if reading z-sections.')
        readMode = 'section';
    else
        assert(size(orig, 1) == 3, 'Origin has to be 3xN-element vector. Is %dx%d-element vector.', size(orig, 1), size(orig, 2));
        assert(size(width, 1) == 3, 'Width has to be 3xN-element vector. Is %dx%d-element vector.', size(width, 1), size(width, 2));
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
            tomomi = orig - r;
            tomoma = orig + (r-1);
            
        case 'cornerwidth'
            tomomi = orig;
            tomoma = orig + width - 1;
            
        case 'section'
            tomomi = zeros(3, numel(orig));
            tomomi(3, :) = orig;
            tomoma = repmat([dim(1) dim(2) 1], 1, numel(orig)]);
    end
    
    % Check that at least 1 pixel is inside the tomogram
    if any(tomomi > dim)
        error('Upper left corner at [%d %d %d]. Cannot exceed [%d %d %d].', ...
              tomomi(1), tomomi(2), tomomi(3), dim(1), dim(2), dim(3));
    end
    
    if any(tomoma < 1)
        error('Lower right corner at [%d %d %d]. Cannot be below [1 1 1].', ...
              tomoma(1), tomoma(2), tomoma(3));
    end
    
    % Get memmap
    mmap = artia.em.memmap(fileName);

    % Enclosure function 
    idx = 1;
    function part = next()
    % Read and pad
    %for i = 1:size(pos, 2)
        % Tomo coordinate
        tmi = tomomi(:, idx);
        tma = tomoma(:, idx);
        %tmi = pos(:, idx) - r;
        %tma = pos(:, idx) + (r-1);
        
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
        part = mmap.Data.data(tmi(1):tma(1), ...
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
                padval = mean(part(:));
            case 'zero'
                padval = 0;
        end
        
        % Pad
        part = padarray(part, pmi, padval, 'pre');
        part = padarray(part, pma, padval, 'post');
        
        idx = idx +1;
    end
    
    iter = @next;
end