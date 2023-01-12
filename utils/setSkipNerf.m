function setSkipNerf(dlnet, cond)
layers = dlnet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer')
        layer.h.structure.(layer.Name).skipNerf = cond;
    end
end

end

