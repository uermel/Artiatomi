function omotl = scale(imotl, scalePos, scaleTrans)
    
    imotl(8:10, :) = imotl(8:10, :) .* scalePos;
    imotl(11:13, :) = imotl(11:13, :) .* scaleTrans;
    
    omotl = imotl;
end

