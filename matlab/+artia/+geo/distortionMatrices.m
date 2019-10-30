function [Sh2, Ro2, St, Ro1, Sh1] = distortionMatrices(angle, amount, dimX, dimY)
% artia.geo.distortionMatrices computes the affine transformations
% used for correction or induction of magnification anisotropy.
%

    % "shiftCenter" matrix in C++
    Sh1 = [1,       0,   -dimX/2;
           0,       1,   -dimY/2;
           0,       0,         1];
    
    % "shiftBack" matrix in C++
    Sh2 = [1,       0,     dimX/2;
           0,       1,     dimY/2;
           0,       0,          1];
       
    % "stretch" matrix in C++
    St =  [amount,  0,            0;
           0,       1,            0;
           0,       0,            1];
       
    % "rotMat1" in C++ 
    angle = deg2rad(angle);
    cosa = cos(angle);
    sina = sin(angle);
    Ro1 = [cosa,    -sina,      0;
           sina,     cosa,      0;
           0,           0,      1];
    
    % "rotMat2" in C++
    cosa = cos(-angle);
    sina = sin(-angle);
    Ro2 = [cosa,    -sina,      0;
           sina,     cosa,      0;
           0,           0,      1];

end