classdef KNNLayer < nnet.layer.Layer & nnet.layer.Formattable & GlobalStruct

    methods
        function layer = KNNLayer()
            % Create a GlobalTFLayer.
            % Set layer name.
            layer.Name = 'KNNLayer';
            % Set layer description.
            layer.Description = "Use knn to predict output";
            % Set layer type.
            layer.Type = "KNNLayer";

            layer.h.structure.Mdl = [];
            layer.h.structure.features = [];
            layer.h.structure.targets = [];

        end

        function addImage(layer, img, T)
            layer.h.structure.features = cat(2,layer.h.structure.features, img);
            layer.h.structure.targets = cat(3, layer.h.structure.targets, T);
        end

        function Z = predict(layer, X)
            % X: image
            if isempty(layer.h.structure.Mdl)
                features = layer.h.structure.features;
                layer.h.structure.Mdl = KDTreeSearcher(features', 'Distance', 'cityblock');
            end
            mdl = layer.h.structure.Mdl;
            IdxNN = knnsearch(mdl, gather(extractdata(X))', 'K', 1);
            Z = dlarray(layer.h.structure.targets(:,:,IdxNN), 'SSB');

        end
    end
end
