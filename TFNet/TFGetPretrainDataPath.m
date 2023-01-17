function [fileNames, file_path] = TFGetPretrainDataPath()
    current_file_path = regexp(which('createPretrainData.m'), '^(.*\/)', 'match');
    tmp = dir(fullfile(current_file_path{1}, 'data_pretrain'));
    tmp = arrayfun(@(x) x.name, tmp, 'UniformOutput', false);
    inds = cellfun(@(x) ~strcmp(x, '..') && ~strcmp(x, '.'), tmp);
    fileNames = tmp(inds);
    file_path = fullfile(current_file_path, 'data_pretrain');
end