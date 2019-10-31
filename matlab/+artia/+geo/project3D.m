function [ali, model] = project3D(xyz, ali, phi, reconDim, imageDim, voxelSize, volumeShifts, magAnisoAmount, magAnisoAngle, refinementShifts)
% artia.geo.project3D projects coordinates from binned, volume-shifted tomograms to
% magnification distorted 2D-Image coordinates.
%
% Parameters:
%   xyz (double):
%       3D positions of particles in the tomogram.
%   ali (double):
%       The EmSART tilt series alignment parameters.
%   phi (double):
%       The angle of beam declination (degrees).
%   reconDim (double[3]):
%       The reconstruction dimensions.
%   imageDim (double[2]):
%       The projection dimensions.
%   voxelSize (double):
%       The voxelsize used for reconstruction.
%   volumeShifts (double[3]):
%       The volumeShifts applied during reconstruction.
%   magAnisoAmount (double):
%       The magnification anisotropy factor (major axis/minor axis, i.e. ellipticity).
%   magAnisoAngle (double):
%       The angle of the minor axis to the x axis (degrees).
%   refinementShifts (double[]):
%       Alignment shifts determined by EmSARTRefine.
%
% Returns:
%   ali (double):
%       The EmSART tilt series alignment parameters (for each input particle). 
%   model (double):
%       The real space model equivalent of the input particles.
%
% Author:
%   UE, 2019
%

    % Binned, shifted tomogram coordinates to unbinned, unshifted coordinates
    %  --> Center points and remove binning
    %  --> Need to add -0.5 because: -1 for C++ -> Matlab, +0.5 for middle
    %      of voxel
    %motive_list(8,:)
    %motive_list(9,:)
    %motive_list(10,:)
    %tx = (motive_list(8,:) - 0.5 + motive_list(11,:) - reconDim(1)/2) * voxelSize - volumeShifts(1);
    %ty = (motive_list(9,:) - 0.5 + motive_list(12,:) - reconDim(2)/2) * voxelSize - volumeShifts(2);
    %tz = (motive_list(10,:)- 0.5 + motive_list(13,:) - reconDim(3)/2) * voxelSize - volumeShifts(3);
    %tx = (x(8,:) - 0.5 + motive_list(11,:) - reconDim(1)/2) * voxelSize - volumeShifts(1);
    %ty = (y(9,:) - 0.5 + motive_list(12,:) - reconDim(2)/2) * voxelSize - volumeShifts(2);
    %tz = (z(10,:)- 0.5 + motive_list(13,:) - reconDim(3)/2) * voxelSize - volumeShifts(3);

    
    % Rotate 90 degrees and mirror along new y-axis to transform to world
    % coordinates used for 3D alignment   
    %x = ty';
    %y = -tx';
    %z = tz';
    
    model = recon2real(xyz, reconDim, voxelSize, volumeShifts);
    x = model(1, :)';
    y = model(2, :)';
    z = model(3, :)';
    
    % Angles 
    thetas = squeeze(ali(1,:,1));
    psis = squeeze(ali(10,:,1));
    
    % Magnification change
    mags = squeeze(ali(9,:,1));
    
    % Shifts
    shiftX = squeeze(ali(5,:,1));
    shiftY = squeeze(ali(6,:,1));

    % Counts
    MarkerCount = numel(x);
    ProjectionCount = numel(thetas);

    % Helper values
    cpsis=cosd(psis);
    spsis=sind(psis);
    cphi=cosd(phi);
    sphi=sind(phi);
    cthetas=cosd(thetas);
    sthetas=sind(thetas);
    
    % Temporary storage
    posU = zeros(ProjectionCount, MarkerCount);
    posV = zeros(ProjectionCount, MarkerCount);
    
    posUo = zeros(ProjectionCount, MarkerCount);
    posVo = zeros(ProjectionCount, MarkerCount);

    for projection = 1 : ProjectionCount
        
        % Get helper values for this projection
        cpsi = cpsis(projection);
        spsi = spsis(projection);
        ctheta = cthetas(projection);
        stheta = sthetas(projection);
        mag = mags(projection);
    
            %m = [[(cphi^2 * cpsi + sphi * ( sphi * cpsi * ctheta + spsi * stheta))/shrink (sphi * cpsi * stheta - spsi * ctheta)/shrink (-(sphi * cpsi * (ctheta - 1) + spsi * stheta) * cphi)/shrink];
            %    [(cphi^2 * spsi + sphi * ( sphi * spsi * ctheta - cpsi * stheta))/shrink (sphi * spsi * stheta + cpsi * ctheta)/shrink (-(sphi * spsi * (ctheta - 1) - cpsi * stheta) * cphi)/shrink]];
        
        % Compute projection matrix
        m11 = (cphi^2 * cpsi + sphi * ( sphi * cpsi * ctheta + spsi * stheta))/mag;
        m12 = (sphi * cpsi * stheta - spsi * ctheta)/mag;
        m13 = (-(sphi * cpsi * (ctheta - 1) + spsi * stheta) * cphi)/mag;

        m21 = (cphi^2 * spsi + sphi * ( sphi * spsi * ctheta - cpsi * stheta))/mag;
        m22 = (sphi * spsi * stheta + cpsi * ctheta)/mag;
        m23 = (-(sphi * spsi * (ctheta - 1) - cpsi * stheta) * cphi)/mag;
            
        for marker = 1 : MarkerCount
            
            % Projected point x
            px =   (m11 * x(marker)) ...
                 + (m12 * y(marker)) ...
                 + (m13 * z(marker)) ...
                 + shiftX(projection) ...
                 + imageDim(1)/2 ...
                 - refinementShifts(marker, projection, 1);
             
            % Projected point x
            pxo =   (m11 * x(marker)) ...
                 + (m12 * y(marker)) ...
                 + (m13 * z(marker)) ...
                 + shiftX(projection) ...
                 + imageDim(1)/2;
             
            % Projected point y
            py =   (m21 * x(marker)) ...
                 + (m22 * y(marker)) ...
                 + (m23 * z(marker)) ...
                 + shiftY(projection) ...
                 + imageDim(2)/2 ...
                 - refinementShifts(marker, projection, 2);
             
             pyo =   (m21 * x(marker)) ...
                 + (m22 * y(marker)) ...
                 + (m23 * z(marker)) ...
                 + shiftY(projection) ...
                 + imageDim(2)/2;
            
            % Check that the points are within the image
            xwithin = px > 0 && px < imageDim(1);
            ywithin = py > 0 && py < imageDim(2);
            
            % Mark points outside invalid
            if xwithin && ywithin
                % Distort the image coordinates
                dist = artia.geo.induceMagAniso([px py], magAnisoAmount, magAnisoAngle, imageDim);
                disto = artia.geo.induceMagAniso([pxo pyo], magAnisoAmount, magAnisoAngle, imageDim);
                
                % Save the distorted points
                posU(projection, marker) = dist(1);
                posV(projection, marker) = dist(2);
                
                posUo(projection, marker) = disto(1);
                posVo(projection, marker) = disto(2);
                
            else
                posU(projection, marker) = -1;
                posV(projection, marker) = -1;
                
                posUo(projection, marker) = -1;
                posVo(projection, marker) = -1;
            end
        
            %posU(itilt, marker) = posU(itilt, marker) + shiftX(projection) / shrink + imdim(1) * 0.5 / shrink;
            %posV(itilt, marker) = posV(itilt, marker) + shiftY(projection) / shrink + imdim(2) * 0.5 / shrink;
        end
    end
    
    % Write new Positions to Marker-File
    ali = zeros(10, ProjectionCount, MarkerCount);
    ali(2,:,:) = posU(:,:);
    ali(3,:,:) = posV(:,:);
    
    ali(7,:,:) = posUo(:,:);
    ali(8,:,:) = posVo(:,:);
    
    % Copy the rest of the parameters
    for projection = 1:ProjectionCount
        ali(1, projection, :) = thetas(projection);
        ali(5, projection, :) = shiftX(projection);
        ali(6, projection, :) = shiftY(projection);
        ali(9, projection, :) = mags(projection);
        ali(10, projection, :) = psis(projection);
    end
    
    % Make the model
    %model = [x'; y'; z'];
    
    % Done!
end

