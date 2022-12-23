classdef KNNLayer < nnet.layer.Layer & nnet.layer.Formattable & GlobalStruct

    methods
        function layer = KNNLayer(name)
            % Create a GlobalTFLayer.
            % Set layer name.
            layer.Name = name;%'KNNLayer';
            % Set layer description.
            layer.Description = "Use knn to predict output";
            % Set layer type.
            layer.Type = "KNNLayer";

            layer.h.structure.([layer.Name 'Mdl']) = [];
            layer.h.structure.([layer.Name 'features']) = [];
            layer.h.structure.([layer.Name 'targets']) = [];

        end

        function addImage(layer, img, T)
            tmp = layer.h.structure.([layer.Name 'features']);
            tmp = cat(2,tmp, img);
            layer.h.structure.([layer.Name 'features']) = tmp;
            tmp = layer.h.structure.([layer.Name 'targets']);
            tmp = cat(3, tmp, T);
            layer.h.structure.([layer.Name 'targets']) = tmp;end

        function Z = predict(layer, X)
            % X: image
            if isempty(layer.h.structure.([layer.Name 'Mdl']))
                features = layer.h.structure.([layer.Name 'features']);
                layer.h.structure.([layer.Name 'Mdl']) = KDTreeSearcher(features', 'Distance', 'cityblock');
            end
            mdl = layer.h.structure.([layer.Name 'Mdl']);
            IdxNN = knnsearch(mdl, gather(extractdata(X))', 'K', 1);
            targets = layer.h.structure.([layer.Name 'targets']);
            Z = dlarray(targets(:,:,IdxNN), 'SSB');

        end
    end
end
