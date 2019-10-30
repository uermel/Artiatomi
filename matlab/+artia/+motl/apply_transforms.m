function outMotl = apply_transforms(transforms, inMotl)

    % File
    inMotl = nameOrFile(inMotl, 'em');
    
    % Apply transforms in order
    for j = 1:numel(transforms)
        % Reset shifts 
        inMotl(8:10, :) = inMotl(8:10, :) + inMotl(11:13, :);
        inMotl(11:13, :) = 0;
        
        shift = transforms(j).shifts;
        rotation = transforms(j).angles;
        
        % Apply additional rotation
        M2 = euler2matrix(rotation);
        for i = 1:size(inMotl, 2)
            angles = inMotl(17:19, i);
            M1 = artia.geo.euler2matrix(angles);
            
            M3 = M2 * M1;

            [phi, psi, theta] = artia.geo.matrix2euler(M3);
            inMotl(17:19, i) = [phi, psi, theta];
        end
        
        % Shift particles by rotation of negative shift vector in reference
        % orientation
        shift = -reshape(shift, 3, 1);
        for i = 1:size(inMotl, 2)
            angles = inMotl(17:19, i);
            M = artia.geo.euler2matrix([-angles(2) -angles(1) -angles(3)]);
            
            pshift = M * shift;

            inMotl(8:10, i) = inMotl(8:10, i) + pshift;
        end
    end
    
    % Output
    outMotl = inMotl;
    integerPos = round(outMotl(8:10, :));
    remainder = outMotl(8:10, :) - integerPos;
    outMotl(8:10, :) = integerPos;
    outMotl(11:13, :) = remainder;

    % Done!
end