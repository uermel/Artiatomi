function fmt = header_fmt()
% artia.em.header_fmt returns a matlab struct with fields corresponding to
% the header sections in the EM-format.
% 
% Usage: 
%   fmt = artia.em.header_fmt();
%
% Returns:
%   fmt (struct):
%       Matlab structure describing the header sections of an EM-file. The
%       fields are ordered by the order of the header. Each field is a cell
%       array containing the byte offset, number of values and encoding for
%       that header section.
%
% Author:
%   Utz H. Ermel 2019

    fmt = struct();
    % Header elements in their order. Order is preserved in MATLAB structures, 
    %so it's important they're not reordered.
    
    % Name                  % Offset[B] % #Values   % Encoding
    fmt.machineCoding =     {0,         1,          'int8',         6};
    fmt.notUsed1 =          {1,         1,          'int8',         0};
    fmt.notUsed2 =          {2,         1,          'int8',         0};
    fmt.dataType =          {3,         1,          'int8',         5};
    fmt.dimX =              {4,         1,          'int32',        0};
    fmt.dimY =              {8,         1,          'int32',        0};
    fmt.dimZ =              {12,        1,          'int32',        0};
    fmt.comment =           {16,        80,         '*char',        zeros(80, 1)};
    fmt.voltage =           {96,        1,          'int32',        0};
    fmt.cs =                {100,       1,          'int32',        0};
    fmt.aperture =          {104,       1,          'int32',        0};
    fmt.magnification =     {108,       1,          'int32',        0};
    fmt.postMagnification = {112,       1,          'int32',        0};
    fmt.exposureTime =      {116,       1,          'int32',        0};
    fmt.objectPixelSize =   {120,       1,          'int32',        0};
    fmt.microscope =        {124,       1,          'int32',        0};
    fmt.pixelsize =         {128,       1,          'int32',        0};
    fmt.CCDArea =           {132,       1,          'int32',        0};
    fmt.defocus =           {136,       1,          'int32',        0};
    fmt.astigmatism =       {140,       1,          'int32',        0};
    fmt.astigmatismAngle =  {144,       1,          'int32',        0};
    fmt.focusIncrement =    {148,       1,          'int32',        0};
    fmt.countsPerElectron = {152,       1,          'int32',        0};
    fmt.intensity =         {156,       1,          'int32',        0};
    fmt.energySlitwidth =   {160,       1,          'int32',        0};
    fmt.energyOffset =      {164,       1,          'int32',        0};
    fmt.tiltangle =         {168,       1,          'int32',        0};
    fmt.tiltaxis =          {172,       1,          'int32',        0};
    fmt.isNewHeaderFormat = {176,       1,          'int32',        0};
    fmt.aliScore =          {180,       1,          'float32',      0};
    fmt.beamDeclination =   {184,       1,          'float32',      0};
    fmt.markerOffset =      {188,       1,          'int32',        0};
    fmt.magAnisoFactor =    {192,       1,          'float32',      0};
    fmt.magAnisoAngle =     {196,       1,          'float32',      0};
    fmt.imageSizeX =        {200,       1,          'int32',        0};
    fmt.imageSizeY =        {204,       1,          'int32',        0};
    fmt.Fillup1 =           {208,       48,         'int8',         zeros(48, 1)};
    fmt.Username =          {256,       20,         '*char',        zeros(20, 1)};
    fmt.Date =              {276,       8,          '*char',        zeros(8, 1)};
    fmt.Fillup2 =           {284,       228,        'int8',         zeros(228, 1)};
end