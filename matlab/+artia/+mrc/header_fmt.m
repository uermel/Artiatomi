function fmt = header_fmt()
% artia.mrc.header_fmt returns a matlab struct with fields corresponding to
% the header sections in the MRC-format.
% 
% Usage: 
%   fmt = artia.MRC.header_fmt();
%
% Returns:
%   fmt (struct):
%       Matlab structure describing the header sections of an EM-file. The
%       fields are ordered by the order of the header. Each field is a cell
%       array containing the byte offset, number of values and encoding for
%       that header section.
%
% Author:
%   UE, 2019

    fmt = struct();
    % Header elements in their order. Order is preserved in MATLAB structures, 
    %so it's important they're not reordered.


    % Name                  % Offset[B] % #Values   % Encoding       %Default
    % Dimensions - 1-12
    fmt.nx =                 {0,         1,          'int32',         0};
    fmt.ny =                 {4,         1,          'int32',         0};
    fmt.nz =                 {8,         1,          'int32',         0};
    
    % Datatype - 13-16
    fmt.mode =               {12,        1,          'int32',         2};
    % writing the datatype  
    %0==1byte, (char/uint8) 
    %1==2byte, (short/int16)
    %2==4byte, (float/float32)
    %3==4byte, (complex short) (NOT SUPPORTED)
    %6==2byte, (uint16)
    
    % Starting point of sub-image - 17-28
    fmt.nxstart =            {16,        1,          'int32',         0};
    fmt.nystart =            {20,        1,          'int32',         0};
    fmt.nzstart =            {24,        1,          'int32',         0};
    
    % Grid size - 29-40
    fmt.mx =                 {28,        1,          'int32',         0};
    fmt.my =                 {32,        1,          'int32',         0};
    fmt.mz =                 {36,        1,          'int32',         0};
    
    % Cell size - 41-52
    fmt.xlen =               {40,        1,          'float32',       0};
    fmt.ylen =               {44,        1,          'float32',       0};
    fmt.zlen =               {48,        1,          'float32',       0};
    
    % Cell angles - 53-64
    fmt.alpha =             {52,         1,          'float32',       90};
    fmt.beta =              {56,         1,          'float32',       90};
    fmt.gamma =             {60,         1,          'float32',       90};
    
    % Index for spacing - 65-76
    fmt.mapc =              {64,         1,          'int32',         1};
    fmt.mapr =              {68,         1,          'int32',         2};
    fmt.maps =              {72,         1,          'int32',         3};
    
    % Min, max, mean - 77-88
    fmt.amin =              {76,         1,          'float32',       0};
    fmt.amax =              {80,         1,          'float32',       0};
    fmt.amean =             {84,         1,          'float32',       0};
    
    % Space group (0 for stack, 1 for volume) - 89-92
    fmt.ispg =              {88,         1,          'int32',         0};
    
    % Number of bytes in ext. header - 93-96
    fmt.next =              {92,         1,          'int32',         1024*32*4};
    
    % Irrelevant - 97-104
    fmt.createid =          {96,         1,          'int16',         0};
    fmt.extra1 =            {98,         6,          'uint8',         zeros(6, 1)};
  
    % Extended header type - 105-108
    fmt.extType =           {104,        4,          '*char',         'FEI1'};
   
    % MRC format type - 109-112
    fmt.nversion =          {108,        1,          'int32',         0};
    
    % Irrelevant - 113 - 128
    fmt.extra2 =            {112,       16,          'uint8',         zeros(16, 1)};
    
    % Extended header spec - 129-132
    fmt.nint =              {128,        1,          'uint16',        0};
    fmt.nreal =             {130,        1,          'uint16',        32};
    
    % Irrelevant - 133-160
    fmt.extra3 =            {132,       28,          'uint8',         zeros(28, 1)};
    
    % Type of data 161-195
    fmt.idtype =            {160,        1,          'int16',         0};
    fmt.lens =              {162,        1,          'int16',         0};
    fmt.nd1 =               {164,        1,          'int16',         0};
    fmt.nd2 =               {166,        1,          'int16',         0};
    fmt.vd1 =               {168,        1,          'int16',         0};
    fmt.vd2 =               {170,        1,          'int16',         0};
    
    fmt.tiltangles =        {172,        6,          'float32',       zeros(6, 1)};
    
    % Origin 197-208
    fmt.xorg =              {196,        1,          'float32',       0};  
    fmt.yorg =              {200,        1,          'float32',       0};
    fmt.zorg =              {204,        1,          'float32',       0};
    
    % MAP statement + machine stamp 209-224
    fmt.cmap =              {208,        4,          '*char',         'MAP '};
    fmt.stamp =             {212,        1,          'uint32',        16708};
    fmt.rms =               {216,        1,          'float32',       1};
    fmt.nlabel =            {220,        1,          'int32',         0};
    
    % Labels 225-1024
    fmt.labels =            {224,      800,          'uchar',         zeros(800, 1)};
    
    % Extended 1025-(1024+fmt.next)
    fmt.extended =          {1024,   32768,          'float32',       zeros(1024*32, 1)};
end
