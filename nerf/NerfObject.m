classdef NerfObject < handle
    % NerfObject wrapper

    properties
    end
    properties(Access=private)
        server
        T
        scale
    end

    methods
        % server1 = ZMQ_Server(5559, 100, 'nerf_box');
        % server2 = ZMQ_Server(5561, 100, 'nerf_cup');
        % server3 = ZMQ_Server(5563, 100, 'nerf_background');
        function obj = NerfObject(name)
            map = containers.Map( ...
                {'nerf_box', 'nerf_cup', 'nerf_background', 'nerf_box2'}, ...
                { 5559, 5561, 5563, 5565});
            port = map(name);
            obj.server = ZMQ_Server(port, 1, name);
            pause(.5)
            obj.T = eye(4);
            obj.testConnection();

            obj.scale = 1.0;
            if ~strcmp(name, 'nerf_background')
                obj.scale =.5;
            end



        end

        function setTransform(obj, T)
            S = eye(4);
            S(1,1) = obj.scale;
            S(2,2) = obj.scale;
            S(3,3) = obj.scale;
            S(4,4) = 1/obj.scale;
            T = T*S;
            obj.T = T;
        end

        function [img, depth] = render(obj, w, h, fov_x)
            obj.renderNonBlock(w, h, fov_x);
            [img, depth] = obj.blockUntilResp();
        end

        function renderNonBlock(obj, h, w, fov_x)
            %             out = obj.server.recv();
            arr = cat(1, reshape(obj.T(1:3,:)',[], 1), w, h, fov_x);
            obj.server.send(arr);
        end

        function [img, depth] = blockUntilResp(obj)
            %             ind = 0;
            %             while ~obj.server.hasNewMsg()
            %                 ind = ind+1;
            %             end

            out = [];
            while isempty(out)
                out = obj.server.recv();
            end

            %             if ~isempty(obj.server.recv())
            %                 keyboard
            %             end
            %             depth = 1;
            %             img = 1;
            depth = out(:,:,5);
            img = out(:,:,1:3);
        end

        function testConnection(obj)
            obj.render(20, 20, 70);
        end
    end
end