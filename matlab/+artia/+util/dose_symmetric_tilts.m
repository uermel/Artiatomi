function tilts = dose_symmetric_tilts(countPerSide, increment, direction)
    tilts = [];
    
    currentAng = 0;
    %direction = 1;
    
    for i = 1:countPerSide
        currentAng = -currentAng;
        tilts = [tilts currentAng];
        currentAng = currentAng + direction * increment;
        tilts = [tilts currentAng];
        direction = -direction;
    end
    currentAng = -currentAng;
    tilts = [tilts currentAng];
end