function out=rot(varargin)
%ROT rotates 2D or 3D images
%  OUT=ROT(IM,[ANGLES],INTERPOLATION,[CENTER])
%  
%  Syntax: out = rot(in,[phi,psi,theta],ip,[center]);
%          out = 3D image output volume
%          in = 3D image input volume
%          phi, psi, theta = Euler angles 
%	   ip = interpolation type: 'linear', 'splines'
%          center = center of the rotation
%  For maximum performance use rot2d or rot3d depending on the
%  dimensions input image
%  In order to rotate XX deg around tilt axis (y-axis) use: 
%                      out=rot(in,[90,-90,XX]);
% The convention: Rotation around z, rotation around x, rotation around z

%AF

switch(nargin)
    
case 0
    error('Error: Not enough input arguments');
    
case 1
    error('Error: Not enough input arguments');
    
case 2    
    im=varargin{1};
    angles=varargin{2};
    interp='linear';
    center=single(floor(size(im)/2)+1);
    [s1,s2,s3]=size(im);
    %s1=size(im,1); s2=size(im,2); s3=size(im,3);
    %disp(center);
    
    if s3==1 & s2>0 & s1>0
        phi=angles(1);
        out = single(zeros(size(im)));
        artia.img.rot2dc(single(im),out,phi,interp,center);
        out = double(out);
    elseif s3>1 & s2>0 & s1>0
        phi=angles(1); psi=angles(2); theta=angles(3);
        out = single(zeros(size(im)));
        artia.img.rot3dc(single(im),out,phi,psi,theta,interp,center);
        out = double(out);        
    end

    
case 3
    im=varargin{1};
    angles=varargin{2};
    interp=varargin{3};
    %center=single(double(size(im))/2); Previously used center
    %Bug corrected Achilleas 5-2-1975
    center=single(floor(size(im)/2)+1);
    s1=size(im,1);
    s2=size(im,2);
    s3=size(im,3);
    if s3==1 & s2>1 & s1>1
        phi=angles(1);
        out = single(zeros(size(im)));
        artia.img.rot2dc(single(im),out,phi,interp,center);
        out = double(out);
    elseif s3>1 & s2>1 & s1>1
        phi=angles(1); psi=angles(2); theta=angles(3);
        out = single(zeros(size(im)));
        artia.img.rot3dc(single(im),out,phi,psi,theta,interp,center);
        out = double(out);        
    end


    
case 4
    im=varargin{1};
    angles=varargin{2};
    interp=varargin{3};
    center=single(varargin{4});
    im=varargin{1};
    angles=varargin{2};
    interp='linear';
    [s1,s2,s3]=size(im);
    if s3==1 & s2>0 & s1>0
        phi=angles(1);
        out = single(zeros(size(im)));
        artia.img.rot2dc(single(im),out,phi,interp,center);
        out = double(out);
    elseif s3>0 & s2>0 & s1>0
        phi=angles(1); psi=angles(2); theta=angles(3);
        out = single(zeros(size(im)));
        artia.img.rot3dc(single(im),out,phi,psi,theta,interp,center);
        out = double(out);        
    end
end



 