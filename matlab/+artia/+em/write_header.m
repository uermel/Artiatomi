function fid = write_header(header, file)
    
    if ischar(file)
        fid = fopen(file,'w','ieee-le');
    else
        fid = file;
    end
    
    %fid = fopen(file,'w','ieee-le');
    
    header_fmt = artia.em.header_fmt();
    names = fieldnames(header_fmt);
    
    for i = 1:numel(names)
        fwrite(fid, header.(names{i}), header_fmt.(names{i}){3});
    end
    
    
     % Close file if filename was supplied
    if ischar(file)
        fclose(fid);
        fid = [];
    end
    %fclose(fid);
end