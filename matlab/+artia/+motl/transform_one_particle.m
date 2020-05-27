function outMotl = transform_one_particle(transforms, inMotl, particleNum)
% artia.motl.transform_one_particle transforms the orientation and 
% translations of a specific particle in a particle list given a list of 
% successive transformations on the reference volume. To apply the same set
% of transformations to all particles in the list, use 
% artia.motl.apply_transforms.
%
% Parameters:
%   transforms (struct):
%       Matlab struct describing the transformations.
%   inMotl (str/double[20xN]):
%       Path to particle list file or particle list matrix.
%   particleNum (integer):
%       Index of the particle within the motivelist
%
% Returns:
%   outMotl (double[20xN]):
%       The particle list with transformed
%       coordinates/orientations/translations.
%
% Author:
%   KS, 2020

    % File
    if ischar(inMotl)
        inMotl = emread(inMotl);
    else
        inMotl = inMotl;
    end

    
    % Apply transforms in order
    for j = 1:numel(transforms)
        % Reset shifts 
        inMotl(8:10, :) = inMotl(8:10, :) + inMotl(11:13, :);
        inMotl(11:13, :) = 0;
        
        shift = transforms(j).shifts;
        rotation = transforms(j).angles;
        
        % Apply additional rotation
        M2 = euler2matrix(rotation);
        angles = inMotl(17:19, particleNum);
        M1 = artia.geo.euler2matrix(angles);

        M3 = M2 * M1;

        [phi, psi, theta] = artia.geo.matrix2euler(M3);
        inMotl(17:19, particleNum) = [phi, psi, theta];
        
        % Shift particles by rotation of negative shift vector in reference
        % orientation
        shift = -reshape(shift, 3, 1);
        angles = inMotl(17:19, particleNum);
        M = artia.geo.euler2matrix([-angles(2) -angles(1) -angles(3)]);

        pshift = M * shift;

        inMotl(8:10, particleNum) = inMotl(8:10, particleNum) + pshift;
    end
    
    % Output
    outMotl = inMotl;
    integerPos = round(outMotl(8:10, :));
    remainder = outMotl(8:10, :) - integerPos;
    outMotl(8:10, :) = integerPos;
    outMotl(11:13, :) = remainder;

    % Done!
end