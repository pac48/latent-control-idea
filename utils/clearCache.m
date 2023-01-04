function clearCache(dlnet)
layers = dlnet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer')
        layer.clearCache();
        if isfield(layer.h.structure, layer.Name)
            layer.h.structure = rmfield(layer.h.structure, layer.Name);
        end
    end
end

end

