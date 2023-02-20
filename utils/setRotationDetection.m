function setRotationDetection(dlnet, value)
layers = dlnet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer')
        layer.h.structure.enableRotation = value;
    end
end

end

