
 
function [meuler, meuler2]=euler2matrix(varargin)
% euler2matrix
% Orthogonal matrix corresponding to a set of rotation parameters
%
% INPUT
%     phi,psi,theta: euler angels
%
% OUTPUT
%     meuler:    the corresponding 3x3 matrix
%
% SYNTAX 
%     meuler=euler2matrix(phi,psi,theta);
%     meuler=euler2matrix([phi,psi,theta]);
%
% NOTE: Angle convention used in rot:
%            rot(v,[phi,psi,theta]) means 
%
%            first:  a rotation of PHI degrees around OLD Z axis
%            second: a rotation of THETA degrees around OLD Y axis
%            third:  a rotation of PSI degrees around OLD Z axis
%
% 
%  In view of this, the set [-psi,-phi,-theta] represents the rotation
%  that inverts the one represented by [phi,psi,theta]
%

if length(varargin)==1
    phi=varargin{1}(1);
    psi=varargin{1}(2);
    theta=varargin{1}(3);
end
    
if length(varargin)==3
    phi=varargin{1};
    psi=varargin{2};
    theta=varargin{3};
end

psi=psi*pi/180;
phi=phi*pi/180;
theta=theta*pi/180;
cospsi=cos(psi);
cosphi=cos(phi);
costheta=cos(theta);
sinpsi=sin(psi);
sinphi=sin(phi);
sintheta=sin(theta);
meuler(1,1)=cospsi*cosphi-costheta*sinpsi*sinphi;
meuler(1,2)=sinpsi*cosphi+costheta*cospsi*sinphi;
meuler(1,3)=sintheta*sinphi;
meuler(2,1)=-cospsi*sinphi-costheta*sinpsi*cosphi;
meuler(2,2)=-sinpsi*sinphi+costheta*cospsi*cosphi;
meuler(2,3)=sintheta*cosphi;
%meulerlast row
meuler(3,1)=sintheta*sinpsi;
meuler(3,2)=-sintheta*cospsi;
meuler(3,3)=costheta;



R_theta=[[costheta 0 sintheta];[0,1,0];[-sintheta,0,costheta]];
R_phi=[[cosphi,sinphi,0];[-sinphi,cosphi,0];[0,0,1]];
R_psi=[[cospsi,sinpsi,0];[-sinpsi,cospsi,0];[0,0,1]];


meuler2=R_psi*R_theta*R_phi;

 