classdef LoFTR < handle
    % Instr wrapper

    properties
    end
    properties(Access=private)
        server
    end

    methods
        function obj = LoFTR()
            obj.server = ZMQ_Server(5557, 10, 'loftr');
            obj.testConnection();

        end

        function [mkpts0, mkpts1, mconf] = predict(obj, image0, image1)
            img = cat(3, rgb2gray(image0), rgb2gray(image1));
            out = double(obj.blockRecv(img));
            if ~isempty(out)
                mkpts0 = out(:, 1:2);
                mkpts1 = out(:,3:4);
                mconf= out(:,5);
            else
                mkpts0 = [];
                mkpts1 = [];
                mconf = [];
            end
        end

        function out = blockRecv(obj, img)
            obj.server.send(img);
            while ~obj.server.hasNewMsg()
            end
            out = obj.server.recv();
        end
        function testConnection(obj)
            img = ones(480, 640, 2, 'uint8')*126;
            obj.blockRecv(img);
        end
    end
end