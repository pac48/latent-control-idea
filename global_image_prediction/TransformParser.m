classdef TransformParser < handle
    % Nerf wrapper

    properties(Access=private)
        nerfObjs
        name2FrameMap
        name2FileMap
        objects_file_path
    end

    methods
        function obj = TransformParser()
            obj.name2FrameMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
            current_file_path = regexp(which('TransformParser.m'), '^(.*\/)', 'match');
            tmp = dir(fullfile(current_file_path{1}, 'objects'));
            tmp = arrayfun(@(x) x.name, tmp, 'UniformOutput', false);
            inds = cellfun(@(x) ~strcmp(x, '..') && ~strcmp(x, '.'), tmp);
            objNames = tmp(inds);
            obj.objects_file_path = fullfile(current_file_path, 'objects');
            paths = cellfun(@(x) fullfile(obj.objects_file_path, x, 'transforms.json'), objNames);
            tmp = cellfun(@(x) jsondecode(fileread(x)).frames, paths, 'UniformOutput', false);
            frames = cellfun(@(x) arrayfun(@(y) y.transform_matrix, x, 'UniformOutput', false)  , tmp  , 'UniformOutput'  , false);

            function out = stripPath(x)
                parts = split(x,'/');
                out = parts(end);
            end
            files = cellfun(@(x) arrayfun(@(y) stripPath(y.file_path), x, 'UniformOutput', false)  , tmp  , 'UniformOutput'  , false);

            keys = cellfun(@(x) x, objNames, 'UniformOutput', false);
            obj.name2FileMap  = containers.Map(keys , files);
            obj.name2FrameMap = containers.Map(keys , frames);

        end

        function [allT, allImgs] = name2Frame(obj, name)
            allT = obj.name2FrameMap(name);
            allNames = obj.name2FileMap(name);
            allImgs = cellfun(@(x) imread(fullfile(obj.objects_file_path{1}, name, x{1})), allNames, 'UniformOutput', false);
        end

    end
end