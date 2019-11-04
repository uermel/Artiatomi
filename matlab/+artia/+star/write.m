function write(instruct, file)
% artia.star.write writes values provided in a matlab structure to a
% star-file.
%
% Usage:
%   e2r_struct2star(file, struct)
%
% Parameters:
%   instruct (struct):
%       struct containing the data to be written. Data will be
%       written with the precision supplied (int32, double or
%       string)
%   file (str):          
%       Path to the output STAR-file
%
% Author:
%   UE, 2019
%

    %%% Get field names and generate format string
    names = fields(instruct);
    format = '';
    for i = 1:numel(names)
        type = class(instruct.(names{i}));
        switch type
            case 'int32'
                format = [format '%d'];
            case 'double'
                format = [format '%f'];
            case 'cell'
                format = [format '%s'];
        end
        
        if i < numel(names)
            format = [format '\t'];
        else
            format = [format '\n'];
        end
    end
    
    %%% Write header
    fid = fopen(file, 'w');
    fprintf(fid, '\n');
    fprintf(fid, 'data_\n\n');
    fprintf(fid, 'loop_\n');
    for i = 1:numel(names)
        fprintf(fid, '_rln%s #%d\n', names{i}, i);
    end
    
    %%% Write data
    for i = 1:numel(instruct.(names{1}))
        line = cell(numel(names), 1);
        for j = 1:numel(names)
            if iscell(instruct.(names{j}))
                line{j} = instruct.(names{j}){i};
            else
                line{j} = instruct.(names{j})(i);
            end
        end
        fprintf(fid, format, line{:});
    end
    
    fclose(fid);
end