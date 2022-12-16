classdef Nerf < handle
    % Nerf wrapper

    properties(Access=private)
        nerfObjs
        nameMap
    end

    methods(Access=private)
        function [img, depth] = removeBackground(obj, img, depth)
            img = img(:,:,1:size(img,3));
            imgBlur = imgaussfilt(img,.5);
            background_ind = any(imgBlur > .04, 3);
            img = img.*background_ind;
            depth = depth.*background_ind;
        end
    end

    methods
        function obj = Nerf(names)
            obj.nameMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
            for i = 1:length(names)
                name = names{i};
                obj.nerfObjs{i} = NerfObject(name);
                obj.nameMap(name) = obj.nerfObjs(i);
            end
        end

        function varargout = render(obj, varargin)
            varargout = cell(nargout, 1);
            if isempty(varargin)
                objs = obj.nerfObjs;
            else
                objs =  cellfun(@(x) obj.nameMap(x), varargin);
            end
            cellfun(@(x) x.renderNonBlock(), objs)
            for i = 1:length(objs)
                [img, depth] = objs{i}.blockUntilResp();
                [img, depth] = obj.removeBackground(img, depth);
                varargout{i} = cat(3, img, depth);
            end
        end
    end
end