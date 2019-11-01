function wedge = dose_weighted(markerfile, tiltOrder, dosePerFrame, pixelsizeInA, volsize)
% artia.wedge.dose_weighted creates exposure dose weighted wedge files using
% the 3D tilt series alignment and tilt order. Tilt axis is y-axis.
%
% Parameters:
%   markerfile (double[10xMxN]):
%       3D tilt series alignment for M projections and N fiducials. Usually
%       marker.ali from a markerfile struct.
%   tiltOrder (double[M]):
%       Tilt angles in order of acquisition.
%   dosePerFrame (double):
%       Dose per projection in e/A^2
%   pixelsizeInA (double):
%       Voxel size of the reconstruction
%   volsize (double):
%       Box size of the particle
%
% Returns:
%   wedge (double[volsize x volsize x volsize]):
%       The wedge file.
%
% Author:
%   MK, UE, 2018

    wedge = zeros(volsize, volsize, volsize);
    thickness = 1.0;
    markerfile = markerfile(:,:,1);
    [~, tiltIdx] = sort(tiltOrder);
    
    accumulatedDose = (0:max(size(tiltOrder,1),size(tiltOrder,2))-1) * dosePerFrame;

    for tilt = 1:max(size(tiltOrder,1),size(tiltOrder,2))
        if (markerfile(2,tilt) > 0)
            tiltAngle = markerfile(1,tilt);
            disp(tiltAngle);
            tiltAngle = tiltAngle / 180 * pi;
            dose = accumulatedDose(tiltIdx(tilt));
            disp(dose);
            %create wedge plane
            %as in RELIION:
            xs = volsize * pixelsizeInA;
            
            for z = 0:volsize-1
                for y = 0:volsize-1
                    for x = 0:volsize-1
                        %int idx = z * size * size + y * size + x;

                        %as in RELIION:
                        posx = x - volsize / 2;
                        posy = y - volsize / 2;
                        posz = z - volsize / 2;

                        temp = posx;

                        posx = cos(tiltAngle) * posx - sin(tiltAngle) * posz;
                        posz = sin(tiltAngle) * temp + cos(tiltAngle) * posz;


                        if (posz < thickness && posz > -thickness) 
                            dist = ((posx / xs) * (posx / xs)) + ((posy / xs) * (posy / xs));
                                                  
                            weight = exp(-dose * dist) * cos(tiltAngle);

                            newVal = min(wedge(x+1, y+1, z+1) + weight, 1.0);
                            wedge(x+1, y+1, z+1) = newVal;
                        end         
                    end
                end
            end
        end
    end
end

