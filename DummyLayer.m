classdef DummyLayer < nnet.layer.Layer & nnet.layer.Formattable
    methods
        function layer = DummyLayer(name)
            % Create a DummyLayer.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Pass input through with no operation";

            % Set layer type.
            layer.Type = "DummyLayer";

        end

        function Z = predict(layer, X)
            Z = X;
        end

    end
end
