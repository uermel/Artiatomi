function [ctfM, CC, defl, defh, astm, defAng] = extern2emsart(defU, defV, defAng, CC)
% artia.ctf.extern2emsart converts CTF parameters in typical RELION
% convention (as returned by GCTF or CTFFind) to EmSART specification.
%
% Specifically, in EmSART convention, the angle is the cw rotation of the
% major astigmatism axis from the image x-axis. The defocus is always
% stored in two values, the minimum and maximum defoci, in that order. This
% is the same convention as in GCTF/CTFFind, but returned values aren't
% always in the order of minimum/maximum (i.e. defU/defV do not always
% contain either minimum or maximum. Thus, in some cases the values need to be switched
% and the angle has to be incremented by 90 degrees. This function does so
% for a single parameter vector. Batched functions are available.
%
% 
    ctfM = zeros(1, 5);

    if defU < defV
        defl = defU/10;
        defh = defV/10; 
    else
        defl = defV/10;
        defh = defU/10;
        defAng = defAng + 90;
    end
    
    defAng = deg2rad(defAng);
    astm = defh-defl;
end