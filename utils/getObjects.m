function objects = getObjects(dlnet)
objects = {};
ind = 0;
for layer = dlnet.Layers'
    if isa(layer, 'NerfLayer')
        ind = ind + 1;
        objects{ind} = layer.objectName;
    end
end
end

