function [newmotl,removedparticles]=deduplicate(motl, tolerance)
% artia.motl.deduplicate removes duplicated particles from a particle list.
% 
% Parameters:
%   motl (str/double[20xN]):
%       Path to the particle list or particle list matrix.
%   tolerance (double):
%       Tolerance radius.
%
% Returns:
%   newmotl (double[20xM]):
%       Deduplicated particle list.
%   removedparticles (doule[20xK]):
%       Removed particles.
%
% Author:
%   ASF
%

if ischar(motl)
    motl = emread(motl);
else
    motl = motl;
end

newmotl=motl;
removedparticles=zeros(0);
cnt=1;
for i=1:size(motl,2)
    if(isempty(find(removedparticles==i, 1))==1)
        checkoverlap=[motl(8,i)+motl(11,i),motl(9,i)+motl(12,i),motl(10,i)+motl(13,i)];
        for j=(i+1):size(motl,2)
            %check overlap
            point=[motl(8,j)+motl(11,j),motl(9,j)+motl(12,j),motl(10,j)+motl(13,j)];
            
            dir=point-checkoverlap;
            len=sqrt(dot(dir,dir));
            if(len<tolerance)
                if (motl(1,i)>motl(1,j))
                    newmotl(:,j)=0;
                    removedparticles(cnt)=j;
                else
                    newmotl(:,i)=0;
                    removedparticles(cnt)=i;
                    j=size(motl,2)+1;
                end
                
                cnt=cnt+1;
            end
        end
    end
    %if rem(i,20)==0 disp(i); end;
end
removedparticles=unique(removedparticles);
newmotl(:,sum(abs(newmotl))==0)=[];
disp([num2str(size(removedparticles,2)) ' removed']);


