classdef ConstLayer < nnet.layer.Layer & nnet.layer.Formattable & GlobalStruct
    properties(Learnable)
        w
    end
    methods
        function layer = ConstLayer(name, outputSize)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Leanable constant";

            % Set layer type.
            layer.Type = "ConstLayer";

            layer.w = zeros(outputSize);

            layer.h.structure.curInd = 1;

        end

        function Z = predict(layer, X)
            assert(numel(X)==1)
%             layer.h.structure.curInd = floor(X);
            global curInd
            curInd = floor(X);
            Z = dlarray(layer.w(:,:,X), 'SSB'); % rpyxyz x allT x batch
        end

    end
end
