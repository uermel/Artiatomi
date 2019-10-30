function ali = imod2emsart(imod_rootname)
% artia.geo.artia2emsart converts an IMOD tomographic tilt series alignment
% to the EmSART convention and format. Needs the .xf and .tlt files.
%
%
    fid = fopen([imod_rootname '.xf']);
    fx = fscanf(fid, '%f %f %f %f %f %f\n', [6, inf]);
    fclose(fid);
    fx = fx';

    fid = fopen([imod_rootname '.tlt']);
    tilt = fscanf(fid, '%f\n', [1, inf]);
    fclose(fid);
    tilt = tilt';


    projCount = size(fx, 1);

    ali = zeros(10, projCount);

    for i = 1:projCount
        a = [fx(i,1) fx(i,2); fx(i,3) fx(i,4)];
        ta = (atan2(-a(1,1), a(1,2)) + atan2(-a(2,2), -a(2,1)))/2 / pi * 180;
        lx = sqrt(sum(([1 0] * a).^2));
        ly = sqrt(sum(([0 1] * a).^2));
        sx = fx(i,1)*fx(i,5) + fx(i,3)*fx(i,6);
        sy = fx(i,2)*fx(i,5) + fx(i,4)*fx(i,6);

        %marker(6, i) = cos(ta)*fx(i,5) + sin(ta)*fx(i,6);
        %marker(5, i) = -sin(ta)*fx(i,5) + cos(ta)*fx(i,6);
        ali(6, i) = -fx(i,5);
        ali(5, i) = fx(i,6);
        ali(5, i) = -sx / lx;
        ali(6, i) = -sy / lx;

    %     res(i,3) = lx;
    %     res(i,4) = ly;
        ali(2,i) = 1;
        ali(3,i) = 1;
        ali(1,i) = -tilt(i);
        ali(8,i) = lx;
        ali(9,i) = ly;
        ali(10,i) = ta-180;
    end
end
