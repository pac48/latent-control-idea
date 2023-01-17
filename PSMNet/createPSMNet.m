function psmNet = createPSMNet(allObjects)
maxBatchSize = 100;

lgraph = layerGraph();
for i = 1:length(allObjects)
    name = allObjects{i};
    if strcmp('background', name)
        continue
    end
    featureInput1 = featureInputLayer(6, 'Name', ['T' name '_cam']);
    lgraph = lgraph.addLayers(featureInput1);

    featureInput2 = featureInputLayer(6, 'Name', ['Trobot_grasp' name]);
    lgraph = lgraph.addLayers(featureInput2);

    tfAllignLayer = TFAllignLayer([name '_TFAllignLayer'], maxBatchSize);
    lgraph = lgraph.addLayers(tfAllignLayer);

   lgraph = lgraph.connectLayers(featureInput1.Name, [tfAllignLayer.Name '/in1']);
   lgraph = lgraph.connectLayers(featureInput2.Name, [tfAllignLayer.Name '/in2']);

end

psmNet = dlnetwork(lgraph);

end