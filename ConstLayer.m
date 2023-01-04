classdef ConstLayer < nnet.layer.Layer & nnet.layer.Formattable
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

        end

        function Z = predict(layer, X)
            Z = dlarray(layer.w, 'SCB');
        end

    end
end
