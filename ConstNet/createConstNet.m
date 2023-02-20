function constNet = createConstNet(allObjects, imageSize, fov)
global nerf

lgraph = layerGraph();
imageInput = imageInputLayer([imageSize(1:2) 3],'name','image_input', 'Normalization','none');
featureInput = featureInputLayer(1, 'Name', 'feature_input');
feature_Trobot_cam2_layer_name = 'Trobot_cam2_input';
feature_Trobot_cam2_layer = featureInputLayer(16, 'Name', feature_Trobot_cam2_layer_name);

lgraph = lgraph.addLayers(imageInput);
lgraph = lgraph.addLayers(featureInput);
lgraph = lgraph.addLayers(feature_Trobot_cam2_layer);


image_layer_name = imageInput.Name;
ind_layer_name = '';
feature_layer_name = featureInput.Name;

for object = allObjects
    name = object{1};
    lgraph = addNerfLayers(lgraph, nerf, {['nerf_' name]}, imageSize, ...
        [name '_nerf_'], image_layer_name, ind_layer_name, feature_layer_name, feature_Trobot_cam2_layer_name);

    TFDummyName = [name '_nerf_T_world_2_cam'];
    TOffset_T_Name = [name '_nerf_TFOffsetLayer/T'];
    dummyLayer = DummyLayer(TFDummyName);
    lgraph = lgraph.addLayers(dummyLayer);
    lgraph = lgraph.connectLayers(TOffset_T_Name, TFDummyName);
end

constNet = dlnetwork(lgraph);
save('constNet', 'constNet', '-v7.3')

end