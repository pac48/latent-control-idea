function constNet = resetConstNet(constNet, allObjects)

lgraph = constNet.layerGraph;
newLayers = [];
l = 1;
while l <= length(lgraph.Layers)
    layer = lgraph.Layers(l);
    if isa(layer, 'ConstLayer')
        lgraph = lgraph.removeLayers(layer.Name);
        layerNew = ConstLayer(layer.Name, size(layer.w));
        newLayers = cat(1, newLayers, layerNew); 
    else
        l = l+1;
    end
end

for i = 1:length(newLayers)
    lgraph = lgraph.addLayers(newLayers(i));
end

for object = allObjects
    name = object{1};
    constLayerName = [name '_nerf_ConstLayer'];
    TFLayerName = [name '_nerf_TFOffsetLayer/in1'];
    lgraph = lgraph.connectLayers(constLayerName, TFLayerName);
    lgraph = lgraph.connectLayers('feature_input', constLayerName);
end

constNet = dlnetwork(lgraph);
% clearCache(constNet)

end