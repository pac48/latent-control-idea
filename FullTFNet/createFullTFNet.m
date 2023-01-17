function fullTFNet =  createFullTFNet(constNet, TFNet)
lgraph = constNet.layerGraph;

feature_layer_name = 'feature_input';
lgraph = removeLayers(lgraph, feature_layer_name);
l = 1;
while l <= length(lgraph.Layers)
    layer = lgraph.Layers(l);
    if isa(layer, 'ConstLayer')
        lgraph = lgraph.removeLayers(layer.Name);
    else
        l = l+1;
    end
end
l = 1;
while l <= length(lgraph.Layers)
    layer = lgraph.Layers(l);
    if isa(layer, 'TFOffsetLayer')
        lgraph = lgraph.removeLayers(layer.Name);
    else
        l = l+1;
    end
end

TFlgraph = TFNet.layerGraph;
allObjects = getObjects(constNet);
for object = allObjects
    name = object{1};
    TFLayerOutName = [name '_TF_FC_Layer'];

    dummyLayer1 = DummyLayer([name '_nerf_T_world_2_cam']);
    dummyLayer2 = DummyLayer([name '_TF']);
    TFlgraph = TFlgraph.addLayers(dummyLayer1);
    TFlgraph = TFlgraph.addLayers(dummyLayer2);
    TFlgraph = TFlgraph.connectLayers(TFLayerOutName, dummyLayer1.Name);
    TFlgraph = TFlgraph.connectLayers(TFLayerOutName, dummyLayer2.Name);
end

for l = 1:length(TFlgraph.Layers)
    layer = TFlgraph.Layers(l);
    if ~isa(layer, 'DummyLayer') % || contains(layer.Name, 'T_world_2_cam')
        lgraph = lgraph.addLayers(layer);
    end
end


for object = allObjects
    name = object{1};
    TFname = [name '_TF_FC_Layer'];
    Scalename = [name 'scaleLayer'];
    Nerfname = [name '_nerf_NerfLayer/in1'];
    TFLayername = [name '_nerf_TFLayer/in2'];
    TFDummyName = [name '_nerf_T_world_2_cam'];
    %     PointCamName = [name '_nerf_TFLayer/points_cam'];
    %     DummyPointCamName = [name '_nerf_TFLayer/points_cam'];

    %             dummyLayer1 = DummyLayer([name '_nerf_T_world_2_cam']);
    %         dummyLayer2 = DummyLayer([name '_TF']);
    %         lgraph = lgraph.addLayers(dummyLayer1);
    %         lgraph = lgraph.addLayers(dummyLayer2);

    lgraph = lgraph.connectLayers(feature_layer_name, Scalename);
    lgraph = lgraph.connectLayers(Scalename, TFname);
    lgraph = lgraph.connectLayers(TFname, Nerfname);
    lgraph = lgraph.connectLayers(TFname, TFLayername);

    lgraph = lgraph.connectLayers(TFname, TFDummyName);

end
% close all
% plot(lgraph)

fullTFNet = dlnetwork(lgraph);

end