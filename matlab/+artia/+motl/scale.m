function omotl = scale(imotl, scalePos, scaleTrans)
% artia.motl.scale rescales the reconstruction coordinates and translation 
% vectors in a particle list according to two factors. 
%
% Parameters:
%   imotl (double[20xN]):
%       Particle list for N particles.
%   scalePos (double):
%       Scaling factor for particle position in the tomogram.
%   scaleTrans (double):
%       Scaling factor for translation vectors.
%
% Returns:
%   omotl (double[20xN]):
%       Rescaled particle list.
% 
% Author:
%   UE, 2019
%

    imotl(8:10, :) = imotl(8:10, :) .* scalePos;
    imotl(11:13, :) = imotl(11:13, :) .* scaleTrans;
    
    omotl = imotl;
end

