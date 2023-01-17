function TFNet = createTFNet(featureNet, allObjects)

out = featureNet.predict(dlarray(zeros(10,10,3),'SSCB'));
numFeatures = size(out,1);

lgraph = layerGraph();
featureInput = featureInputLayer(numFeatures, 'Name', 'feature_input');
lgraph = lgraph.addLayers(featureInput);
for object = allObjects
    name = object{1};
    TFLayerOut = fullyConnectedLayer(6, "Name", [name '_TF_FC_Layer']);
    scaleLayer = functionLayer(@(x) .001*x, 'Name', [name 'scaleLayer']);
    lgraph = lgraph.addLayers(TFLayerOut);
    lgraph = lgraph.addLayers(scaleLayer);
    lgraph = lgraph.connectLayers(featureInput.Name, scaleLayer.Name);
    lgraph = lgraph.connectLayers(scaleLayer.Name, TFLayerOut.Name);
end

TFNet = dlnetwork(lgraph);

end