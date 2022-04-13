function [meuler, meuler2]=euler2matrix(angles)
% artia.em.euler2matrix computes orthogonal matrix corresponding to a set of 
% rotation parameters
%
% Parameters:
%   angles (double[3]):
%       [phi,psi,theta] (euler angles).
%
% Returns:
%   meuler (double[3x3]):    
%       the corresponding rotation matrix (phi, theta, psi, ZYZ convention)
%
% Author:
%   DCD, UE

phi = angles(1);
psi = angles(2);
theta = angles(3);

psi = psi * pi / 180;
phi = phi * pi / 180;
theta = theta * pi / 180;
cospsi = cos(psi);
cosphi = cos(phi);
costheta = cos(theta);
sinpsi = sin(psi);
sinphi = sin(phi);
sintheta = sin(theta);
meuler(1, 1) =  cospsi * cosphi - costheta * sinpsi * sinphi;
meuler(1, 2) =  sinpsi * cosphi + costheta * cospsi * sinphi;
meuler(1, 3) =  sintheta * sinphi;
meuler(2, 1) = -cospsi * sinphi - costheta * sinpsi * cosphi;
meuler(2, 2) = -sinpsi * sinphi + costheta * cospsi * cosphi;
meuler(2, 3) =  sintheta * cosphi;
%meulerlast row
meuler(3, 1) =  sintheta * sinpsi;
meuler(3, 2) = -sintheta * cospsi;
meuler(3, 3) =  costheta;



R_theta=[[costheta 0 sintheta];[0,1,0];[-sintheta,0,costheta]];
R_phi=[[cosphi,sinphi,0];[-sinphi,cosphi,0];[0,0,1]];
R_psi=[[cospsi,sinpsi,0];[-sinpsi,cospsi,0];[0,0,1]];


meuler2=R_psi*R_theta*R_phi;

 