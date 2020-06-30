function [average] = extract_write(motivelistFilename, upScaleFactor, tomogramFilenameArray, r, doRotate, doTranslate, doNormalize, partPref)
%extractParts creates a cell-array of sub-tomograms from a motivelist and 
%and an array of tomogram filenames and averages the particles
%
% Usage:
%   [cellPart, average] = extractParts(motivelistFilename, upScaleFactor, tomogramFilenameArray, r, noRotation, noTranslation, noNormalization);
%
% Parameters:
%   motivelistFilename (str/double) 
%       Motivelist filename or matrix (Entries of row 5 specify cell of tomo 
%       filename array). See example below.
%   upScaleFactor (double):
%       Multiplier for coordinates and shifts.
%   tomogramFilenameArray
%       array of tomogram filenames (Filenames should
%       be placed at the index corresponding to the entry 
%       in row 5 of the motivelist. (i.e. for motl(5, i) == 4, 
%       the filename should be placed at tomogramFilenameArray{4}). 
%       See example below. 
%   r (double):
%       Radius of the box to be extracted.
%   doRotate (bool):
%       If true, particles will be rotated according to motl(17:19).
%   doTranslate (bool):
%       If true, particles will be translated according to motl(11:13).
%   doNormalize
%       If true, mean will be subtracted from each particle (recommended).
%
% Returns:
%   average (double[r*2 x r*2 x r*2]
%       Average of all particles.
%
% Example:
%   .. codeblock matlab
%         % example motivelist
%         motl = zeros(20,5);
% 
%         % example tomogram numbers
%         motl(5, :) = [1 3 5 7 9];
% 
%         % save
%         emwrite(motl, motivelistFilename);
% 
%         % example filename array
%         tomogramFilenameArray = {};
%         tomogramFilenameArray{1} = '/path/to/file1.em';
%         tomogramFilenameArray{3} = '/path/to/file3.em';
%         tomogramFilenameArray{5} = '/path/to/file5.em';
%         tomogramFilenameArray{7} = '/path/to/file7.em';
%         tomogramFilenameArray{9} = '/path/to/file9.em';
% 
%         % run the function
%         [cellPart, average] = extractParts(motivelistFilename, 1, tomogramFilenameArray, 32, 1, 1, 1);
%
% Authors:
%   UE 2018
    
    % Read file if necessary
    if ischar(motivelistFilename)
        motl = artia.em.read(motivelistFilename);
    else
        motl = motivelistFilename;
    end
    
    tomos = unique(motl(5,:));
    numberOfTomograms= numel(tomos);
    
    % Test size of tomo array
    assert(size(tomogramFilenameArray,2) >= max(motl(5,:)), 'Tomogram array smaller than largest TomoNr in Motivelist.')

    fprintf('%g tomograms are processed\n',numberOfTomograms);
    average= zeros(2*r,2*r,2*r);
    
    for t = 1:numel(tomos)

        tomogramFilename = tomogramFilenameArray{tomos(t)};
        tempMotl = motl(:, motl(5, :) == tomos(t));
        npart = size(tempMotl, 2);
        tempMotl(8:16, :) = upScaleFactor * round(tempMotl(8:16, :));

        fprintf('Working on %s\n',tomogramFilename);
        n = 0;

        for i = 1:npart
            if ~doTranslate
                tempMotl(11, i) = 0;
                tempMotl(12, i) = 0;
                tempMotl(13, i) = 0;
            end

            x=tempMotl(8, i)  + tempMotl(11, i);
            y=tempMotl(9, i)  + tempMotl(12, i);
            z=tempMotl(10, i) + tempMotl(13, i);
          
            part = artia.em.read_inc(tomogramFilename, [x y z], [2*r 2*r 2*r]);
            
            if doNormalize
                partMean = mean(part(:));
                partStd = std(part(:));
                part = (part-partMean);
            end

            phi = tempMotl(17, i); 
            psi = tempMotl(18, i); 
            theta = tempMotl(19, i);

            if doRotate
                part = artia.img.rot(part, [-psi, -phi, -theta]);
            end

            artia.em.write(part, sprintf([partPref '%d_%d.em'], tempMotl(5, i), tempMotl(6, i)));
            average = average + part;
            
            msg = sprintf('Read %d/%d particles.\n', i, npart);
            fprintf(repmat('\b',1,n));
            fprintf(msg);
            n=numel(msg);
        end
    end
end

