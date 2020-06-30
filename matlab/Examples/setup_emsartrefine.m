% Example script to set up an EmSARTRefine run, which refines your tomogram
% alignments by locally aligning particles to the given reference

%% Split latest motivelist into motivelists for each tomogram
motl = artia.em.read(...
    '/path/to/your/project/data/averaging/motls/motl_[last_iteration].em');

tomonr = [1, 2, 3, 4, 5];

for i = 1:numel(tomonr)
    idx = motl(5,:)==tomonr(i);
    tomo_motl = motl(:,idx);
    
    % Write out the individual motivelists
    artia.em.write(tomo_motl, ...
        sprintf('/path/averaging/refinement/motls/%d_ref_motl.em', ...
            tomonr(i)));
end

%% Create a refinement reference by overlaying the mask and the latest ref
% Load mask and latest reference
ref = artia.em.read('/path/to/data/average/ref/ref[last_iteration].em');
mask = artia.em.read('/path/to/data/average/other/mask.em');
% Overlay them
ref_mask = (ref .* mask);
avg_ref_mask = mean(ref_mask(:));
std_ref_mask = std(ref_mask(:));

ref_refinement = (ref_mask - avg_ref_mask) ./ std_ref_mask;
artia.em.write(ref_refinement, '/path/average/refinement/ref_refinement');