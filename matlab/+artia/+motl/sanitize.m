function omotl = sanitize(imotl, tomoNr, wedgeNr)
% artia.motl.sanitize prepares tomogram specific particle lists for 
% concatenation to the global particle list.
%
% Parameters:
%   imotl (double[20xN]):
%       Particle list for N particles.
%   tomoNr (int):
%       Tomogram number that the particles originate from.
%   wedgeNr (int):
%       Wedge to use during averaging.
%
% Returns:
%   omotl (double[20xN]):
%       The cleaned particle list.
%
% Author:
%   UE, 2019
%
    
    % These shouldn't be assigned
    imotl([1 2 3], :) = 0;
    
    % Row 5 stores tomo idx
    imotl(5, :) = tomoNr;
    % Row 6 stores particle idx
    imotl(6, :) = 1:size(imotl, 2);
    % Row 7 stores wedge idx
    imotl(7, :) = wedgeNr;
    % These should be empty
    imotl(14:16, :) = 0;
    imotl(20, :) = 1;
    
    omotl = imotl;
end

