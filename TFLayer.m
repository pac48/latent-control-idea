classdef TFLayer < nnet.layer.Layer & nnet.layer.Formattable
    methods
        function layer = TFLayer(name)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = name;%'TFLayer';
            layer.InputNames = {'in1', 'in2'};
            layer.OutputNames = {'points_cam', 'empty'};

            % Set layer description.
            layer.Description = "Transform 3D point from one frame to another";

            % Set layer type.
            layer.Type = "TFLayer";

        end

        function [pointsCam, empty] = predict(layer, points, X)
            % X: T (16 x batch) this is transform from world to camera
            % coordinates
            % points: 3d points (3 x n x batch)
            % Z: 3d points (3 x n x batch)
            empty = dlarray([], 'SSB');
            pointsCam = dlarray([], 'SSB');

            if isempty(X) || isempty(points)
                return
            end

            if size(X, 1) == 6 % convert to T
                X = reshape(X, 6,1,[]);
                T = getT(X(1:3,:), X(4:6,:));
            else
                T = reshape(X, 4, 4, []);
            end
            % T world to cam

            points = extractdata(points(1:3,:,:));
            points = reshape(points, 3, []);
            pointsCam = T(1:3, 1:3, :)*points + T(1:3, end, :);
            pointsCam = dlarray(pointsCam, 'SSB');

        end

    end
end
