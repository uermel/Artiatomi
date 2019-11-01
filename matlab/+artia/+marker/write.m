function write(marker, fileName)

    % Get ali (data) dimension
    [xdim, ydim, zdim] = size(marker.ali);
    
    % Compute 3D model offset (header + ali size)
    offset = 512 + 4 * xdim * ydim * zdim;
    
    % Create EM-header
    header = artia.em.default_header('dimX', xdim, ...
                                     'dimY', ydim, ...
                                     'dimZ', zdim, ...
                                     'isNewHeaderFormat', 1, ...
                                     'aliScore', marker.aliScore, ...
                                     'beamDeclination', marker.beamDeclination, ...
                                     'markerOffset', offset, ...
                                     'magAnisoFactor', marker.magAnisoFactor, ...
                                     'magAnisoAngle', marker.magAnisoAngle, ...
                                     'imageSizeX', marker.imageSizeX, ...
                                     'imageSizeY', marker.imageSizeY);
                                 
    fid = artia.em.write(marker.ali, fileName, 'header', header, 'returnFid', true);   
    fwrite(fid, marker.model, 'float32');
    fclose(fid);
    
    
%     fid = fopen(marker_name,'w','ieee-le');
%     
%     % Write header -- this should always be identical for marker files
%     fwrite(fid, 6, 'uint8'); % machine - LE
%     fwrite(fid, 0, 'uint8'); % not used
%     fwrite(fid, 0, 'uint8'); % not used
%     fwrite(fid, 5, 'uint8'); % data type - float
% 
%     fwrite(fid, xdim, 'uint32'); % dim X
%     fwrite(fid, ydim, 'uint32'); % dim Y
%     fwrite(fid, zdim, 'uint32'); % dim Z
% 
%     fwrite(fid, zeros(80, 1), 'uint8'); % comment
%     fwrite(fid, zeros(20, 1), 'int32'); % header info - not used
%     fwrite(fid, 1, 'int32'); % IsNewHeader
%     fwrite(fid, marker.AliScore, 'float32'); % Alignment score
%     fwrite(fid, marker.Phi, 'float32'); % BeamDeclination
%     fwrite(fid, 512 + 4 * xdim * ydim * zdim, 'int32'); % Marker model offset from file beginning
%     fwrite(fid, marker.MagAniso, 'float32'); % Mag Anisotropy
%     fwrite(fid, marker.MagAnisoAngle, 'float32'); % Mag Anisotropy Angle
%     fwrite(fid, marker.ImSizeX, 'int32'); % Image Size X
%     fwrite(fid, marker.ImSizeY, 'int32'); % Image Size Y
%     fwrite(fid, zeros(12, 1), 'int32'); % Fillup
%     fwrite(fid, zeros(20, 1), 'uint8'); % Username
%     fwrite(fid, zeros(8, 1), 'uint8'); % Date
%     fwrite(fid, zeros(228, 1), 'uint8'); % Fillup
%     
%     % write out data
%     fwrite(fid, marker.ali, 'float32');
%     marker.model(2, :) = -marker.model(2, :);
%     fwrite(fid, marker.model, 'float32');
%     
%     % Finished.
%     fclose(fid);
end