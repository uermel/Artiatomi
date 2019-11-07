function [header, endian] = read_header(fileName)
% artia.mrc.read_header -- Read header of files in MRC-format
%
% Usage:
%    [header, endian] = artia.mrc.read_header(fileName)
%
% Arguments:
%   fileName (str):           
%       File in MRC-Format [character vector]
%
% Returns:
%    header (struct):
%        Matlab struct containing header info
%
%    endian (str):      
%       string indicating endianess
%
%
% Author:
%   UE, 2019  

    % Open file for reading
    fid = fopen(fileName, 'r');
    
    % Figure out byte order. 
    % This is complicated by the fact that certain software writes out
    % big endian data using the little endian machine stamp (e.g. some 
    % versions of Digital Micrograph. Thus, a dirty method of checking the
    % validity of the image X-dimension is used when the extended header
    % type is not stated (which is true for the DM version this was tested with).
    % Allowed types are:
    extTypes = {'CCP4', 'MRCO', 'SERI', 'AGAR', 'FEI1', 'HDF5'};
    
    %Read machine stamp 
    fseek(fid, 212, 'bof');
    bytes = fread(fid, 4, 'char');
    
    % Figure out potential byte order
    if bytes(1) == 68 && ismember(bytes(2), [68, 65])
        potEndian = 'ieee-le';
    elseif bytes(1) == 17 && bytes(2) == 17
        potEndian = 'ieee-be';
    else
       warning(sprintf('Unknown byte order: %d %d %d %d. Trying LE.', bytes))
       potEndian = 'ieee-le';
    end
    fclose(fid);
    
    % Confirm byte order using CMAP statement
    fid = fopen(fileName, 'r', potEndian);
    fseek(fid, 208, 'bof');
    cmap = fread(fid, 4, '*char');
    
    if all(cmap' == 'MAP ')
        % Correct ordering found, go back to beginning of file
        endian = potEndian;
        fseek(fid, 0, 'bof');
    else
        % Unknown extType or wrong order. Use x-dim validation instead. This
        % will fail with files having xdim > 100,000
        fclose(fid);
        modes = {'ieee-le', 'ieee-be'};
        fidLE = fopen(fileName, 'r', 'ieee-le');
        fidBE = fopen(fileName, 'r', 'ieee-be');
        xdimLE = fread(fidLE, 1, 'uint32');
        xdimBE = fread(fidBE, 1, 'uint32');
        endian = modes{[xdimLE xdimBE] < 100000};
        warning('MAP statement not found using encoding derived from machine stamp. Guessing %s based on xdim.', endian);
        fclose(fidLE);
        fclose(fidBE);
    end
    
    %return
    
    fid = fopen(fileName,'r',endian);
    if fid==-1 
        error('Wrong File Name: %s', em_name); 
    end

    % Start header
    header = struct();
    % Dimensions - 1-12
    header.nx = fread(fid, 1, 'int32'); 
    header.ny = fread(fid, 1, 'int32'); 
    header.nz = fread(fid, 1, 'int32'); 
    
    % Datatype - 13-16
    header.mode = fread(fid, 1,'int32'); 
    % writing the datatype  
    %0==1byte, (char/uint8) 
    %1==2byte, (short/int16)
    %2==4byte, (float/float32)
    %3==4byte, (complex short)
    %6==2byte, (uint16)
    
    % Starting point of sub-image - 17-28
    header.nxstart = fread(fid, 1, 'int32');	
    header.nystart = fread(fid, 1, 'int32');	
    header.nzstart = fread(fid, 1, 'int32');
    
    % Grid size - 29-40
    header.mx = fread(fid, 1, 'int32');	
    header.my = fread(fid, 1, 'int32');	
    header.mz = fread(fid, 1, 'int32');
    
    % Cell size - 41-52
    header.xlen = fread(fid, 1, 'float32');	
    header.ylen = fread(fid, 1, 'float32');	
    header.zlen = fread(fid, 1, 'float32');
    
    % Cell angles - 53-64
    header.alpha = fread(fid, 1, 'float32');	
    header.beta = fread(fid, 1, 'float32');	
    header.gamma = fread(fid, 1, 'float32');
    
    % Index for spacing - 65-76
    header.mapc = fread(fid, 1, 'int32');	
    header.mapr = fread(fid, 1, 'int32');	
    header.maps = fread(fid, 1, 'int32');
    
    % Min, max, mean - 77-88
    header.amin = fread(fid, 1, 'float32');	
    header.amax = fread(fid, 1, 'float32');	
    header.amean = fread(fid, 1, 'float32');
    
    % Space group (0 for stack, 1 for volume) - 89-92
    header.ispg = fread(fid, 1, 'int32');	
    
    % Number of bytes in ext. header - 93-96
    header.next = fread(fid, 1, 'int32');
    
    % Irrelevant - 97-104
    header.createid = fread(fid, 1, 'uint16');	  %96
    header.extra1 = fread(fid, 6, 'uint8');	    
    
    % Extended header type - 105-108
    header.extType = fread(fid, 4, '*char')';
    
    % MRC format type - 109-112
    header.nversion = fread(fid, 1, 'int32')/10;
    
    % Irrelevant - 113 - 128
    header.extra2 = fread(fid, 16, 'uint8');
    
    % Extended header spec - 129-132
    header.nint = fread(fid, 1, 'uint16');	 
    header.nreal = fread(fid, 1, 'uint16');	
    
    % Irrelevant - 133-160
    header.extra3 = fread(fid, 28, 'uint8');
    
    % Type of data 161-1024
    header.idtype = fread(fid, 1, 'int16');	 
    header.lens = fread(fid, 1, 'int16');	
    header.nd1 = fread(fid, 1, 'int16');	
    header.nd2 = fread(fid, 1, 'int16');	
    header.vd1 = fread(fid, 1, 'int16');	 
    header.vd2 = fread(fid, 1, 'int16');	
    header.tiltangles = fread(fid, 6, 'float32');	
    header.xorg = fread(fid, 1, 'float32');	
    header.yorg = fread(fid, 1, 'float32');
    header.zorg = fread(fid, 1, 'float32');
    header.cmap = fread(fid, 4, '*char')';	    
    header.stamp = fread(fid, 4, 'char');	  
    header.rms = fread(fid, 1, 'float32');	
    header.nlabel = fread(fid, 1, 'int32');	 
    %ftell(fid)
    header.labels = char(fread(fid, [80 10], 'uchar')');
    if header.nlabel < 10 
        header.labels(header.nlabel+1:end, :) = '';
    end
    %ftell(fid)
    if header.next > 0 && strcmp(header.extType, 'SERI')
        % IMOD Format
        % Bit flag decoding
        info = {'tiltAngle', ...
                'xyzCoords', ...
                'stagePos', ...
                'magnification', ...
                'intensity', ...
                'exposureDose', ...
                'custom4bit1', ...
                'custom4bit2', ...
                'custom2bit1', ...
                'custom2bit2', ...
                'custom2bit3'};
        bytes = [2 6 4 2 2 4 4 4 2 2 2];
        vars = [1 3 2 1 1 1 1 1 1 1 1];
        factor = [100 1 25 0.01 25000 1 1 1 1 1 1];
        shorts = bytes./2;
        flags = [1 2 4 8 16 32 128 512 64 256 1024];
        presentVals = bitand(header.nreal, flags) == flags;
        assert(sum(bytes(presentVals)) == header.nint);
        presentShorts = sum(shorts(presentVals));
        
        % Read the data and reshape by number of shorts
        ext = fread(fid, header.nz * presentShorts, 'int16');
        ext = reshape(ext, presentShorts, []); 
        
        % Decode
        names = info(presentVals);
        nfactor = factor(presentVals);
        nShorts = shorts(presentVals);
        nVars = vars(presentVals);
        
        % Figure out range for values
        starts = nShorts(1);
        ends = nShorts(1);
        for i = 2:numel(nShorts)
            starts(i) = ends(i-1) + 1;
            ends(i) =  starts(i) + nShorts(i) - 1;
        end
        
        % Convert values using factors
        extended = struct();
        for i = 1:numel(names)
            if nVars(i) == numel(starts(i):ends(i)) % Values are shorts
                extended.(names{i}) = ext(starts(i):ends(i), :) ./ nfactor(i);
            else % Values are floats
                s1 = ext(starts(i), :);
                s2 = ext(starts(i), :);
                as1 = abs(s1); 
                as2 = abs(s2);
                extended.(names{i}) = (s1./as1).*(as1.*256 + mod(as2, 256)) .* 2.^((s2/as2) * (as2./256));
            end
        end
        
        header.extended = extended;
        
    elseif header.next > 0 && (strcmp(header.extType, 'FEI1') || header.next == 131072)
        % FEI/Kunz header
        ext = fread(fid, header.next/4, 'float32');
        ext = reshape(ext, 32, 1024);
        names = {'tiltAngle', ...
                'tiltAngleBeta', ...
                'stagePos', ...
                'imageShift', ...
                'defocus', ...
                'expTime', ...
                'meanInt', ...
                'tiltAxis', ...
                'pixelSize', ...
                'magnification', ...
                'remainder'};
        vars = [1 1 3 2 1 1 1 1 1 1 19];   
        starts = vars(1);
        ends = vars(1);
        for i = 2:numel(vars)
            starts(i) = ends(i-1) + 1;
            ends(i) =  starts(i) + vars(i) - 1;
        end
        
        extended = struct();
        for i = 1:numel(names)
            extended.(names{i}) = ext(starts(i):ends(i), :);
        end
        
        header.extended = extended;
        
    elseif header.next > 0
        % Any other format, read as bytes
        ext = fread(fid, header.next, 'uint8');
        header.extended = ext;
    else
        header.extended = [];
    end
    
    % Close file
    fclose(fid);
    
    % Done !
end