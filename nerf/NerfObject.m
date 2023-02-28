classdef NerfObject < handle
    % NerfObject wrapper

    properties
    end
    properties(Access=private)
        server
        T = eye(4)
        scale = 1
    end

    methods
        % server1 = ZMQ_Server(5559, 100, 'nerf_box');
        % server2 = ZMQ_Server(5561, 100, 'nerf_cup');
        % server3 = ZMQ_Server(5563, 100, 'nerf_background');
        function obj = NerfObject(name)
            map = containers.Map( ...
                {'nerf_box', 'nerf_cup', 'nerf_background', 'nerf_book', 'nerf_iphone_box', 'nerf_plate', 'nerf_blue_block', 'nerf_fork', 'nerf_drawer'...
                'nerf_new_plate', 'nerf_jug', 'nerf_napkin', 'nerf_pepper', 'nerf_salt'}, ...
                { 5559, 5561, 5563, 5565, 5567, 5569, 5571, 5573, 5575, 5577, 5579, 5581, 5583, 5585});
            port = map(name);
            obj.server = ZMQ_Server(port, 1, name);
            pause(.5)
            obj.T = eye(4);
            obj.testConnection();

            s1 = 0.1584*(1/.8)*0.9215;
            s2 = 0.1692*(1/.8);
%             s1 = 1.0;
%             s2 = 1.0;

            mapScale = containers.Map( {'nerf_background', 'nerf_book', 'nerf_iphone_box', 'nerf_plate', 'nerf_fork', 'nerf_blue_block', 'nerf_drawer', 'nerf_new_plate', 'nerf_jug', 'nerf_napkin', 'nerf_pepper', 'nerf_salt'}, ...
                {.5*1.25, s1, s1, s1, s1, s1, s2, s2, s2, s2, s2, s2});

            obj.scale = mapScale(name);


        end

        function T = scaleTransform(obj, T)
%             S = eye(4);
%             S(1,1) = obj.scale;
%             S(2,2) = obj.scale;
%             S(3,3) = obj.scale;
%             S(4,4) = 1/obj.scale;
% 
%             T = T*S;
            T(1:3,end) = T(1:3,end).*obj.scale;

        end

        function T = undoScaleTransform(obj, T)
%             S = eye(4);
%             S(1,1) = 1/obj.scale;
%             S(2,2) = 1/obj.scale;
%             S(3,3) = 1/obj.scale;
%             S(4,4) = obj.scale;
% 
%             T = T*S;
            T(1:3,end) = T(1:3,end)./obj.scale;
        end

        function setTransform(obj, T)
            T = obj.undoScaleTransform(T);
            obj.T = T;
        end
% % 
% %         function [img, depth] = render(obj, w, h, fov_x)
% %             %             assert(0, 'not working')
% %             obj.renderNonBlock(w/3, h/3, fov_x/2);
% %             %             obj.renderNonBlock(w, h, fov_x);
% %             [img, depth] = obj.blockUntilResp();
% %             img = imresize(img, [h,w]);
% %             depth = imresize(depth, [h,w], 'nearest');
% %             depth = depth*(1/obj.scale); % maybe inv
% % 
% %         end

        function renderNonBlock(obj, h, w, fov_x)
            %             out = obj.server.recv();
            arr = cat(1, reshape(obj.T(1:3,:)',[], 1), w, h, fov_x/2);
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
            depth = depth*obj.scale; % maybe inv *.8
            img = out(:,:,1:3);

        end

        function testConnection(obj)
            obj.renderNonBlock(20, 20, 70);
            obj.blockUntilResp();
        end
    end
end