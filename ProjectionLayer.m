classdef ProjectionLayer < nnet.layer.Layer & nnet.layer.Formattable

    properties
        fx
        fy
        fl
    end

    methods
        function layer = ProjectionLayer(name, fl, fx, fy)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = name;%'ProjectionLayer';
            % Set layer description.
            layer.Description = "Project 3D point in camera coordinates to pixel coordinates";

            % Set layer type.
            layer.Type = "ProjectionLayer";
            layer.OutputNames = {'mkptsNerf'};


            layer.fx = fx;
            layer.fy = fy;
            layer.fl = fl;

        end

        function out = predict(layer, points)
            % point: 3d points (3 x n x batch)
            % n is the number of detected key points

            X = points(1,:,:);
            Y = points(2,:,:);
            Z = points(3,:,:);
            x = -layer.fl*X./Z;
            y = -layer.fl*Y./Z;


            out = cat(1, x, y);

        end

%         function [dLdX] = backward(layer, X, ~, dLdZ, ~)
% 
%             dLdX = X*0;
% 
%         end
    end
end
