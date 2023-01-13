classdef TFLayer < nnet.layer.Layer & nnet.layer.Formattable
    methods
        function layer = TFLayer(name)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = name;%'TFLayer';
            layer.InputNames = {'in1', 'in2'};
            layer.OutputNames = {'out1', 'points_cam'};

            % Set layer description.
            layer.Description = "Transform 3D point from one frame to another";

            % Set layer type.
            layer.Type = "TFLayer";

        end

        function [Z, pointsCam] = predict(layer, points, X)
            % X: T (16 x batch) this is transform from world to camera
            % coordinates
            % points: 3d points (3 x n x batch)
            % Z: 3d points (3 x n x batch)

            if isempty(X)
                X = eye(4);
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
            Z = pointsCam;
            Z = dlarray(Z, 'SSB');
            pointsCam = dlarray(pointsCam, 'SSB');
            %             mean(points,2)
        end

%         function [dldpoints, dLdX] = backward(layer, points, X, ~, dLdZ, ~)
%             %             Z(1,1) = T11*P(1,1) + T12*P(2,1) + T13*P(3,1) + T14
%             %             Z(2,1) = T21*P(1,1) + T22*P(2,1) + T23*P(3,1) + T14
%             %             Z(3,1) = T31*P(1,1) + T32*P(2,1) + T33*P(3,1) + T34
%             dldpoints = points*0;
%             dLdX = X*0;
%         end

        %
        %
        %             dldpoints = points*0;
        %
        %             dZ1dX1 = points(1,:);
        %             dZ2dX2 = points(1,:);
        %             dZ3dX3 = points(1,:);
        %             dZ3dX4 = points(1,:)*0;
        %
        %             dZ1dX5 = points(2,:);
        %             dZ2dX6 = points(2,:);
        %             dZ3dX7 = points(2,:);
        %             dZ3dX8 = points(2,:)*0;
        %
        %             dZ1dX9 = points(3,:);
        %             dZ2dX10 = points(3,:);
        %             dZ3dX11 = points(3,:);
        %             dZ3dX12 = points(3,:)*0;
        %
        %             dZ1dX13 = points(3,:)*0+1;
        %             dZ2dX14 = points(3,:)*0+1;
        %             dZ3dX15 = points(3,:)*0+1;
        %             dZ3dX16 = points(3,:)*0;
        %
        % %             tmp = reshape(dLdZ, size(dLdZ));
        %             dLdX(1,1) = pagemtimes(dLdZ(1,:), dZ1dX1');
        %             dLdX(2,1) = pagemtimes(dLdZ(2,:), dZ2dX2');
        %             dLdX(3,1) = pagemtimes(dLdZ(3,:), dZ3dX3');
        %             dLdX(4,1) = 0;
        %             dLdX(5,1) = pagemtimes(dLdZ(1,:), dZ1dX5');
        %             dLdX(6,1) = pagemtimes(dLdZ(2,:), dZ2dX6');
        %             dLdX(7,1) = pagemtimes(dLdZ(3,:), dZ3dX7');
        %             dLdX(8,1) = 0;
        %             dLdX(9,1) = pagemtimes(dLdZ(1,:), dZ1dX9');
        %             dLdX(10,1) = pagemtimes(dLdZ(2,:), dZ2dX10');
        %             dLdX(11,1) = pagemtimes(dLdZ(3,:), dZ3dX11');
        %             dLdX(12,1) = 0;
        %             dLdX(13,1) = -pagemtimes(dLdZ(1,:), dZ1dX13');
        %             dLdX(14,1) = pagemtimes(dLdZ(2,:), dZ2dX14');
        %             dLdX(15,1) = pagemtimes(dLdZ(3,:), dZ3dX15');
        %             dLdX(16,1) = 0;
        %
        % %             tmp = reshape(dLdX, 4, 4);
        % %             Tinv = inv(reshape(extractdata(X),4,4));
        % %             dLdX = -Tinv*tmp*Tinv;
        % %             T = reshape(extractdata(X),4,4);
        % %             dLdX = -T*tmp*T;
        %
        % %             dLdX = reshape(dLdX,[], 1);
        %
        %
        %
        % %             dLdX([1:12 14 15],1) = 0;
        %
        % %             dLdX = -dLdX;
        %
        %         end
    end
end
