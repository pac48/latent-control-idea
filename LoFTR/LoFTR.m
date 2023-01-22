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
            w0 = size(image0, 2);
            h0 = size(image0, 1);
            w1 = size(image1, 2);
            h1 = size(image1, 1);
            image0 = imresize(image0, [480 640]*1);
            image1 = imresize(image1, [480 640]*1);

            img = zeros(size(image0, 4), 1, size(image0,1), size(image0,2), 2, 'uint8');
            for i = 1:size(image0, 4)
                img(i, :, :,:,:) = cat(3, rgb2gray(image0(:,:,:,i)), rgb2gray(image1(:,:,:,i)));
            end

            out = double(obj.blockRecv(img));
            if ~isempty(out)
                mkpts0 = out(:, 1:2).*[w0/(640*1) h0/(480*1)];
                mkpts1 = out(:, 3:4).*[w1/(640*1) h1/(480*1)];
                mconf= out(:,5);
            else
                mkpts0 = [];
                mkpts1 = [];
                mconf = [];
            end
        end

        function out = blockRecv(obj, img)
%             out = obj.server.recv();
            obj.server.send(img);
            pause(1/1000);
%             out = [];
            while ~obj.server.hasNewMsg()
%                 out = obj.server.recv();
            end
            out = obj.server.recv();
        end
        function testConnection(obj)
            image0 = ones(480, 640, 3, 'uint8')*126; 
            image1 = ones(480, 640, 3, 'uint8')*126;
            obj.predict(image0, image1)
        end
    end
end