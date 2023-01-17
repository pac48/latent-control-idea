function [fileNames, file_path] = TFGetDataPath()
    current_file_path = regexp(which('createPretrainData.m'), '^(.*\/)', 'match');
    tmp = dir(fullfile(current_file_path{1}, 'data'));
    tmp = arrayfun(@(x) x.name, tmp, 'UniformOutput', false);
    inds = cellfun(@(x) ~strcmp(x, '..') && ~strcmp(x, '.'), tmp);
    fileNames = tmp(inds);
    file_path = fullfile(current_file_path, 'data');
end