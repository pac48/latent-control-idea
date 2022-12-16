classdef NerfObject < handle
    % NerfObject wrapper

    properties
    end
    properties(Access=private)
        server
        T
    end

    methods
        % server1 = ZMQ_Server(5559, 100, 'nerf_box');
        % server2 = ZMQ_Server(5561, 100, 'nerf_cup');
        % server3 = ZMQ_Server(5563, 100, 'nerf_background');
        function obj = NerfObject(name)
            map = containers.Map( ...
                {'nerf_box', 'nerf_cup', 'nerf_background'}, ...
                { 5559, 5561, 5563});
            port = map(name);
            obj.server = ZMQ_Server(port, 10, name);
            pause(.5)
            obj.T = eye(4);
            obj.testConnection();

        end

        function setTransform(obj, T)
            obj.T = T;
        end

        function [img, depth] = render(obj)
            obj.renderNonBlock();
            [img, depth] = obj.blockUntilResp();
        end

        function renderNonBlock(obj)
            obj.server.send(obj.T(1:3,:));
        end

        function [img, depth] = blockUntilResp(obj)
            while ~obj.server.hasNewMsg()
            end
            out = obj.server.recv();
            depth = out(:,:,5);
            img = out(:,:,1:3);
        end

        function testConnection(obj)
            obj.render();
        end
    end
end