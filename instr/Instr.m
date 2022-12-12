classdef Instr < handle
    % Instr wrapper

    properties
    end
    properties(Access=private)
        server
    end

    methods
        function obj = Instr()
            obj.server = ZMQ_Server(5555, 10, 'instr');
            obj.testConnection();

        end

        function segments = predict(obj, left_image, right_image)
            img = cat(2,left_image, right_image);
            img = img(:,:,[3 2 1]);
            segments = double(obj.blockRecv(img));
        end

        function out = blockRecv(obj, img)
            obj.server.send(img);
            while ~obj.server.hasNewMsg()
            end
            out = obj.server.recv();
        end
        function testConnection(obj)
            img = ones(480, 640*2, 3, 'uint8')*126;
            obj.blockRecv(img);
        end
    end
end