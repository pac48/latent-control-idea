function T = getObjectTransforms(dlnet, map, name)
T = [];
key = [name '_nerf_TFOffsetLayer/T'];
if ~isKey(map, key)
    return
end
out = map(key);
for layer = dlnet.Layers'
    if isa(layer, 'NerfLayer')
        lName = [name '_nerf_NerfLayer'];
        if strcmp(layer.Name, lName)
            j = layer.h.structure.(lName).jBest;
            if isempty(j)
                return
            end
            T = out(:,:,j);
            return
        end
    end
end
end

