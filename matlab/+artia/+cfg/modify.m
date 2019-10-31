function modify(inpaths, outpaths, varargin)
% artia.cfg.modify changes entries in a list of Artiatomi config files.
%
% inpaths - cell array of cfg filenames to use as input
% example: inpaths = {'/path/to/infileA.cfg', '/path/to/infileB.cfg'}
% 
% outpaths - cell array of output filenames for the new config files
% example: outpaths = {'/path/to/outfileA.cfg', '/path/to/outfileB.cfg'}
% 
% All entries to be modified are added as name-value pairs.
% In order to change an entry two possibilities exist:
% 
% 1. Change/Append an entry with the same value in all config files. 
%    example: 
%    To set 'VoxelSize' to 4 in all configs in the infiles list:
%
%    modify_configs(inpaths, outpaths, 'VoxelSize', {'4'});
%
%    IMPORTANT: Convert Numbers to strings.
%    IMPORTANT: Single values also have to contained in a cell array.
%
% 2. Change/Append an entry with a different value for each config file.
%    example: 
%    To set 'VoxelSize' to 4 in the first config, to 8 in the second
%    config and to 12 in the third config:
%
%    modify_configs(inpaths, outpaths, 'VoxelSize', {'4', '8', '12'});
%
%    IMPORTANT: Convert Numbers to strings.
%
%
%   See also: cfg2struct, struct2cfg
%
% UE 2018
    params = {'CudaDeviceID', ... 
            'ProjectionFile', ... 
            'OutVolumeFile', ... 
            'MarkerFile', ... 
            'Lambda', ... 
            'Iterations', ... 
            'RecDimesions', ... 
            'UseFixPsiAngle', ... 
            'PsiAngle', ... 
            'PhiAngle', ... 
            'OverSampling', ... 
            'VolumeShift', ... 
            'VoxelSize', ... 
            'DimLength', ... 
            'CutLength', ... 
            'Crop', ... 
            'CropDim', ... 
            'CtfMode', ... 
            'SkipFilter', ... 
            'fourFilterLP', ... 
            'fourFilterLPS', ... 
            'fourFilterHP', ... 
            'fourFilterHPS', ... 
            'SIRTCount', ... 
            'CtfFile', ... 
            'BadPixelValue', ... 
            'CorrectBadPixels', ... 
            'AddTiltAngle', ... 
            'AddTiltXAngle', ... 
            'CTFBetaFac', ... 
            'FP16Volume', ... 
            'WriteVolumeAsFP16', ... 
            'ProjectionScaleFactor', ... 
            'ProjectionNormalization', ... 
            'WBP', ... 
            'Cs', ... 
            'Voltage', ... 
            'IgnoreZShiftForCTF', ... 
            'CTFSliceThickness', ... 
            'SizeSubVol', ... 
            'VoxelSizeSubVol', ... 
            'MotiveList', ... 
            'Reference', ... 
            'MaxShift', ... 
            'ShiftOutputFile', ... 
            'ShiftInputFile', ...
            'GroupMode', ... 
            'MaxDistance', ... 
            'GroupSize', ... 
            'SpeedUpDistance', ... 
            'CCMapFileName', ... 
            'NamingConvention', ...
            'ScaleMotivelistShift', ... 
            'ScaleMotivelistPosition', ...
            'MagAnisotropy', ...
            'SubVolPath', ...
            'Particles', ...
            'ClearAnglesIteration', ...
            'LowPass', ...
            'HighPass', ...
            'Sigma', ...
            'AngIter', ...
            'PhiAngIter', ...
            'MaskCC', ...
            'PhiAngIncr', ...
            'AngIncr', ...
            'WedgeFile', ...
            'BatchSize', ...
            'WBPFilter', ...
            'WedgeIndices', ...
            'PathWin', ...
            'PathLinux', ...
            'StartIteration', ...
            'EndIteration', ...
            'BestParticleRatio', ...
            'ApplySymmetry', ...
            'MultiReference', ...
            'Classes', ...
            'CouplePhiToPsi', ...
            'RotateMaskCC', ...
            'Mask'};
        
    p = inputParser;
    
    errorMsg = 'Input must be either a single value or the length of the infiles array.'; 
    validationFcn = @(x) assert((size(inpaths, 2) == size(x, 2)) || (size(x, 2) == 1),errorMsg);
    
    for i = 1:size(params, 2)
        paramName = params{i};
        defaultVal = {};
        addParameter(p,paramName,defaultVal, validationFcn);
    end
    
    parse(p, varargin{:});
    results = fieldnames(p.Results);
    
    universalNames = {};
    universalParams = {};
    uniqueNames = {};
    uniqueParams = {};

    for i = 1:numel(results)
        if size(p.Results.(results{i}), 2) ==  1
            universalNames = [universalNames results{i}];
            universalParams = [universalParams p.Results.(results{i})];
        elseif size(p.Results.(results{i}), 2) ==  size(inpaths, 2)
            uniqueNames = [uniqueNames results{i}];
            uniqueParams = [uniqueParams; p.Results.(results{i})];
        end
    end
    
    for i = 1:size(inpaths, 2)
        tempstruct = artia.cfg.read(inpaths{i});
        
        for j = 1:size(universalNames, 2)
            tempstruct.(universalNames{j}) = universalParams{j};
        end
        
        for j = 1:size(uniqueNames, 2)
            tempstruct.(uniqueNames{j}) = uniqueParams{j,i};
        end
        
        artia.cfg.write(tempstruct, outpaths{i});
    end
end