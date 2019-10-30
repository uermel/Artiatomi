function [ wedge ] = dose_weighted(markerfile, tiltOrder, dosePerFrame, pixelsizeInA, volsize)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    %The dose compensation part is taken from RELION. Note that in RELION
    %the accumulated dose is first multiplied by 4, then in the variable K4
    %devided by 4. Here we simply ommit this :)
    %See ctf.h and ctf.cpp of RELION source code.

    wedge = zeros(volsize, volsize, volsize);
    thickness = 1.0;
    markerfile = markerfile(:,:,1);
    [sortedTiltOrder, tiltIdx] = sort(tiltOrder);
    
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

