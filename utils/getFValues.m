function [fl, fx, fy] = getFValues(dlnet)
layers = dlnet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer')
        [fl, fx, fy] = layer.getFValues();
        return
    end
end

end