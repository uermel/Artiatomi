% Example code snippet to clear out the shift values for particles in a
% motivelist. Necessary to set up for re-averaging based on locally refined
% particles from EmSARTRefine and EmSartSubVolumes

motl = artia.em.read('/path/to/data/average/motls/motl_10.em');
motl(11:13, :) = 0;
artia.em.write(motl, '/path/to/data/average/refinement/motls/motl_1.em');