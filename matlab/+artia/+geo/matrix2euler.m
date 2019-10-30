function [varargout] = matrix2euler(rm)
% [phi,psi,theta] = matrix2euler(rotationMatrix)
% Convert a rotation matrix into Euler angles
%
% The order of angles is PHI,PSI,THETA, as used in function rot
% 
%
tol=1e-4;
if and(rm(3,3)>1-tol,rm(3,3)<1+tol)
    warning('indetermination in defining phi and psi: rotation about z');
    theta=0;
    phi=-atan2(rm(2,1),rm(1,1))*180/pi;
    psi=0;
else
    theta=acos(rm(3,3));
    phi=atan2(rm(1,3),rm(2,3));
    psi=atan2(rm(3,1),-rm(3,2));


    theta=theta*180/pi;
    phi=phi*180/pi;
    psi=psi*180/pi;
end

if (nargout)==1
    varargout{1}=[phi,psi,theta];
else
    varargout{1}=phi;
    varargout{2}=psi;
    varargout{3}=theta;
end
