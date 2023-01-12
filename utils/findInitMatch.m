function inds = findInitMatch(dlnet, img, object)
layers = dlnet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer') && strcmp(layer.objectName, object)
        inds = layer.findInitMatch(img);
        break
    end
end
for layer = layers'
    if isa(layer, 'TFOffsetLayer') && strcmp(layer.objectName, object)
        layer.setIndT(inds);
        break
    end
end

end