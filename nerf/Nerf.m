classdef Nerf < handle
    % Nerf wrapper

    properties(Access=private)
        nerfObjs
        nameMap
        name2FrameMap
        name2FileMap
        objects_file_path
    end
    properties
        names
        fov_x = 70.8193;
    end

    methods(Access=private)
        function [img, depth] = removeBackground(obj, img, depth)
            img = img(:,:,1:size(img,3));
%             imgBlur = imgaussfilt(img,.5);
%             background_ind = any(imgBlur > .04, 3);
            background_ind = depth ~= 0;
            img = img.*background_ind;
%             depth = imgaussfilt(depth, .5);
%             background_ind = any(img > .2, 3);
            depth = depth.*background_ind;
           
            mask = depth ~= 0;
            K = ones(3);
            for i = 1:3
            mask = conv2(mask, K,"same") == 9;
%             imshow(mask)
            end
            depth = depth.*mask;

%             inds = depth < .75*mean(depth(depth>0), 'all');
%             depth(inds) = 0;
        end

        function img = colorAdjust(obj, img)
            img = lin2rgb(img);
            low = 0.1;
            high = 1.0;
            img = imadjust(img,[low high],[]); % I is double

        end

    end

    methods
        function obj = Nerf(names)
            obj.names = names;

            obj.nameMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.name2FrameMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
            current_file_path = regexp(which('Nerf.m'), '^(.*\/)', 'match');
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

            for i = 1:length(names)
                name = names{i};
                obj.nerfObjs{i} = NerfObject(name);
                obj.nameMap(name) = obj.nerfObjs(i);
            end
        end

        function setTransform(obj, varargin)
            for i = 1:length(varargin)
                pair = varargin{i};
                T = pair{2};
                name = pair{1};
                tmp = obj.nameMap(name);
                tmp{1}.setTransform(T);
            end
        end

        function [allT, allImgs] = name2Frame(obj, name)
            allT = obj.name2FrameMap(name);
            allNames = obj.name2FileMap(name);
            allImgs = cellfun(@(x) imread(fullfile(obj.objects_file_path{1}, name, x{1})), allNames, 'UniformOutput', false);
        end

        function [imgRender, imDepth] = render(obj, w, h, fov_x)
            cellfun(@(x) x.renderNonBlock(w, h, fov_x), obj.nerfObjs)
            imgRender = zeros(w, h, 3);
            imDepth = zeros(w, h, 1);
            for i = 1:length(obj.nerfObjs)
                [img, depth] = obj.nerfObjs{i}.blockUntilResp();
                [img, depth] = obj.removeBackground(img, depth);
                img = obj.colorAdjust(img);
                imgRender =  img;
            end

        end

        function varargout = renderObject(obj, h, w, fov_x, varargin)
            varargout = cell(nargout, 1);
            if isempty(varargin)
                objs = obj.nerfObjs;
            else
                objs =  cellfun(@(x) obj.nameMap(x), varargin);
            end
            cellfun(@(x) x.renderNonBlock(h, w, fov_x), objs)
            for i = 1:length(objs)
                [img, depth] = objs{i}.blockUntilResp();
%                 img = imresize(img, [h,w]);
%                 depth = imresize(depth, [h,w], 'nearest');

                inds = depth < 1.0;
                [img, depth] = obj.removeBackground(img, depth);
                depth(inds) = 0;
                img = obj.colorAdjust(img);
                varargout{2*i-1} = img;
                varargout{2*i} = depth;
                % varargout{i} = 1;
            end
        end
    end
end