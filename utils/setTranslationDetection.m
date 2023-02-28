function setTranslationDetection(dlnet, value)
layers = dlnet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer')
        layer.h.structure.enableTranslation = value;
    end
end

end

