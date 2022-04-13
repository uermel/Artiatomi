function c=move(varargin)
%MOVE Moves an image in X,Y,Z direction and fills the rest with zeros
%   C=MOVE(A,[dx dy dz]) This function moves image A in X,Y and Z direction. the
%   coordinates of the first pixel is now (dx+1,dy+1,dz+1). The rest of the image
%   is filled with zeros.
%
%

if nargin<2
    error('Not Enough Input Arguments');
    return;
end
if nargin>2
    error('Too Many Input Arguments');
    return;
end
a=varargin{1};
coord=varargin{2};
%coord=floor(coord); This is option used in EM!!!! This is because of the integer transfer
coord=round(coord); 

[s1 s2 s3]=size(a);
% 2-D Image Case
if s3 == 1    
    if abs(coord(1))>=s1 | abs(coord(2))>=s2 % Out of limits move
        error('Error: Coordinates out of range! Use Different x, y, z...');
        return;
    end
    b=zeros(s1,s2);
    if coord(1)>=0 & coord(2)>=0   
        atmp=a(1:(s1-coord(1)),1:(s2-coord(2)));   
        b((coord(1)+1):s1,(coord(2)+1):s2)=atmp;
        c=b;
    elseif coord(1)<0 & coord(2)>=0
        atmp=a((abs(coord(1))+1):s1,1:(s2-coord(2)));   
        satmp=size(atmp);
        b(1:satmp(1),(coord(2)+1):s2)=atmp;
        c=b;    
    elseif coord(1)>=0 & coord(2)<0
        atmp=a(1:(s1-coord(1)),(abs(coord(2))+1):s2);    
        satmp=size(atmp);
        b((coord(1)+1):s1,1:satmp(2))=atmp;
        c=b;
    else
        atmp=a((abs(coord(1))+1):s1,(abs(coord(2))+1):s2);
        satmp=size(atmp);
        b(1:satmp(1),1:satmp(2))=atmp;
        c=b;
    end
else
    % 3-D Image Case
    if abs(coord(1))>=s1 | abs(coord(2))>=s2 | abs(coord(3))>=s3 % Out of limits move
        error('Wrong Input Arguments');
        return;
    end
    c=zeros(s1,s2,s3);
    b=zeros(s1,s2);
    if coord(3)>=0
        for i=1:(s3-coord(3))
            if coord(1)>=0 & coord(2)>=0   
                atmp=a(1:(s1-coord(1)),1:(s2-coord(2)),i);   
                b((coord(1)+1):s1,(coord(2)+1):s2)=atmp;
                c(:,:,(i+coord(3)))=b;
            elseif coord(1)<0 & coord(2)>=0
                atmp=a((abs(coord(1))+1):s1,1:(s2-coord(2)),i);   
                satmp=size(atmp);
                b(1:satmp(1),(coord(2)+1):s2)=atmp;
                c(:,:,(i+coord(3)))=b;    
            elseif coord(1)>=0 & coord(2)<0
                atmp=a(1:(s1-coord(1)),(abs(coord(2))+1):s2,i);    
                satmp=size(atmp);
                b((coord(1)+1):s1,1:satmp(2))=atmp;
                c(:,:,(i+coord(3)))=b;
            else
                atmp=a((abs(coord(1))+1):s1,(abs(coord(2))+1):s2,i);
                satmp=size(atmp);
                b(1:satmp(1),1:satmp(2))=atmp;
                c(:,:,(i+coord(3)))=b;
            end
        end
    else
        for i=(abs(coord(3))+1):s3
            if coord(1)>=0 & coord(2)>=0   
                atmp=a(1:(s1-coord(1)),1:(s2-coord(2)),i);   
                b((coord(1)+1):s1,(coord(2)+1):s2)=atmp;
                c(:,:,(i-abs(coord(3))))=b;
            elseif coord(1)<0 & coord(2)>=0
                atmp=a((abs(coord(1))+1):s1,1:(s2-coord(2)),i);   
                satmp=size(atmp);
                b(1:satmp(1),(coord(2)+1):s2)=atmp;
                c(:,:,(i-abs(coord(3))))=b;    
            elseif coord(1)>=0 & coord(2)<0
                atmp=a(1:(s1-coord(1)),(abs(coord(2))+1):s2,i);    
                satmp=size(atmp);
                b((coord(1)+1):s1,1:satmp(2))=atmp;
                c(:,:,(i-abs(coord(3))))=b;
            else
                atmp=a((abs(coord(1))+1):s1,(abs(coord(2))+1):s2,i);
                satmp=size(atmp);
                b(1:satmp(1),1:satmp(2))=atmp;
                c(:,:,(i-abs(coord(3))))=b;
            end            
        end
    end
end
