classdef InvLayer < nnet.layer.Layer & nnet.layer.Formattable
    methods
        function layer = InvLayer()
            % Create a TFLayer.

            % Set layer name.
            layer.Name = 'InvLayer';

            % Set layer description.
            layer.Description = "Apply transform inverse";

            % Set layer type.
            layer.Type = "InvLayer";

        end

        function Z = predict(layer, X)
            % X: T (16 x batch)
            % Z: Tinv (16 x batch)

            T = reshape(X, 4, 4, []);
%             Tinv = inv(extractdata(T));
            R = T(1:3, 1:3)';
            p = -R*T(1:3,end);
            Tinv = [[R p];[0 0 0 1]];

            Z = dlarray(reshape(Tinv, 16, []), 'SB');

        end
% 
%         function [dLdX] = backward(layer, X, ~, dLdZ, ~)
%             T = reshape(X, 4, 4, []);
% 
% %             dY = reshape(dLdZ, 4, 4);
% %             Yinv = (extractdata(T));
% % 
% %             dLdX = -Yinv*dY*Yinv;
% % 
%             dLdX = X*0;
%         end

    end
end
