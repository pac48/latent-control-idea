function setDetectObjects(dlnet, objects)
layers = dlnet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer')
        layer.h.structure.detectObjects = objects;
    end
end

end

