classdef TFOffsetLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        T0
    end
    methods
        function layer = TFOffsetLayer(name)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = name; %'TFOffsetLayer';
            
            layer.InputNames = {'in1', 'in2'};
            layer.OutputNames = {'out1', 'out2'};

            % Set layer description.
            layer.Description = "apply constant transform to input";

            % Set layer type.
            layer.Type = "TFOffsetLayer";

%             layer.T0 = T0;

        end

        function T = getT(~, rpy, p)
            ct = cos(rpy);
            st = sin(rpy);
            R11 = ct(1, :).*ct(3, :).*ct(2, :) - st(1, :).*st(3, :);
            R12 = -ct(1, :).*ct(2, :).*st(3, :) - st(1, :).*ct(3, :);
            R13 = ct(1, :).*st(2, :);
            R21 = st(1, :).*ct(3, :).*ct(2, :) + ct(1, :).*st(3, :);
            R22 = -st(1, :).*ct(2, :).*st(3, :) + ct(1, :).*ct(3, :);
            R23 = st(1, :).*st(2, :);
            R31 = -st(2, :).*ct(3, :);
            R32 = st(2, :).*st(3, :);
            R33 = ct(2, :);

            %             T = [R11 R12 R13 p(1)
            %                 R21 R22 R23 p(2)
            %                 R31 R32 R33 p(3)
            %                 0   0   0   1];
            T = [R11; R21; R31; 0; R12; R22; R32; 0; R13; R23; R33; 0; p(1); p(2); p(3); 1];
        end

        function R = getR(~, rpy)
            rpy = reshape(rpy, 3,[]);
            ct = cos(rpy);
            st = sin(rpy);
            R11 = ct(1, :).*ct(3, :).*ct(2, :) - st(1, :).*st(3, :);
            R12 = -ct(1, :).*ct(2, :).*st(3, :) - st(1, :).*ct(3, :);
            R13 = ct(1, :).*st(2, :);
            R21 = st(1, :).*ct(3, :).*ct(2, :) + ct(1, :).*st(3, :);
            R22 = -st(1, :).*ct(2, :).*st(3, :) + ct(1, :).*ct(3, :);
            R23 = st(1, :).*st(2, :);
            R31 = -st(2, :).*ct(3, :);
            R32 = st(2, :).*st(3, :);
            R33 = ct(2, :);

            %             T = [R11 R12 R13 p(1)
            %                 R21 R22 R23 p(2)
            %                 R31 R32 R33 p(3)
            %                 0   0   0   1];
            R = [R11, R12, R13 
                R21, R22, R23
                R31, R32, R33];
        end

        function [Z1, Z2] = predict(layer, X, T0)
            % X: xyz rpy (6 x batch) this is transform from world to camera
            % coordinates
            % Z: transform matrix (16 x batch)
%             X = [.01; .01; .01; .1; .01; .01].*X;
            X = X./10;

%             T = T0; 
%             T(1:3, 1:4) = T(1:3, 1:4) + reshape(X, 3, 4, []);
%             T(1:3, 1) = T(1:3, 1)./sqrt(sum(T(1:3, 1).^2, 1));
%             T(1:3, 2) = T(1:3, 2)./sqrt(sum(T(1:3, 2).^2, 1));
%             T(1:3, 3) = T(1:3, 3)./sqrt(sum(T(1:3, 3).^2, 1));

%             T(1, 1:3) = T(1, 1:3)./sum(sqrt(T(1, 1:3).^2), 2);
%             T(2, 1:3) = T(2, 1:3)./sum(sqrt(T(2, 1:3).^2), 2);
%             T(3, 1:3) = T(3, 1:3)./sum(sqrt(T(3, 1:3).^2), 2);

%             T = layer.getT(X(4:6, :), X(1:3, :));
%             T = reshape(T,4,4,[]);
            T0 = reshape(T0, 4, 4, []);
            R = layer.getR(X(4:6, :));
%             R2 = layer.getR(X(7:9, :)*10);
            R = pagemtimes(T0(1:3, 1:3, :), R);
            
%             R = pagemtimes(R2, R);
            
%             P = reshape(X(1:3, :), 3 ,[]);
%             P = pagemtimes(T0(1:3, 1:3, :), P) + T0(1:3,end);
            P = 10*reshape(X(1:3, :), 3 ,[]) + T0(1:3,end);
            T = [[R P]; 0 0 0 1];

            %             T = [X(1) X(2) X(3) X(4)
            %                 X(5) X(6) X(7) X(8)
            %                 X(9) X(10) X(11) X(12)
            %                 0 0 0 1];
            %             T = T + layer.T0;
            Z = reshape(T, 16, []);
            Z1 = dlarray(Z, 'CB');
            Z2 = dlarray(Z, 'CB');
        end
% % 
%         function [dLdX] = backward(layer, X, ~, dLdZ, ~)
% 
%             dLdX = X*0;
% 
%         end

    end
end
