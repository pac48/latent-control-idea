classdef ReshapeLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        % (Optional) Layer properties.
        OutputSize
    end
    properties (Learnable)
        % Layer learnable parameters.
        
        Weights
        Bias
    end
    
    methods
        function layer = ReshapeLayer(outputSize, name)
            % Create a projectAndReshapeLayer.
            
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "reshape layer with output size " + join(string(outputSize));
            
            % Set layer type.
            layer.Type = "Reshape";
            
            % Set output size.
            layer.OutputSize = outputSize;

        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data, specified as a 1-by-1-by-C-by-N 
            %                 dlarray, where N is the mini-batch size.
            % Outputs:
            %         Z     - Output of layer forward function returned as 
            %                 an sz(1)-by-sz(2)-by-sz(3)-by-N dlarray,
            %                 where sz is the layer output size and N is
            %                 the mini-batch size.

            % Reshape.
            outputSize = layer.OutputSize;
            Z = reshape(X, outputSize(1), outputSize(2), []);
            Z = dlarray(Z, 'SCB');
        end
    end
end
