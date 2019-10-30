function omotl = sanitize(imotl, tomoNr, wedgeNr)
    
    imotl([1 2 3], :) = 0;
    imotl(5, :) = tomoNr;
    imotl(6, :) = 1:size(imotl, 2);
    imotl(7, :) = wedgeNr;
    imotl(14:16, :) = 0;
    imotl(20, :) = 1;
    
    omotl = imotl;
end

