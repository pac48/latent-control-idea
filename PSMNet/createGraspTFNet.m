function graspTFNet = createGraspTFNet(objects)
maxBatchSize = 100;

lgraph = layerGraph();
name = objects;
if strcmp('background', name)
    error('background not needed')
end
featureInput1 = featureInputLayer(16, 'Name', ['T' name '_cam']);
lgraph = lgraph.addLayers(featureInput1);

featureInput2 = featureInputLayer(16, 'Name', ['Trobot_grasp' name]);
lgraph = lgraph.addLayers(featureInput2);

tfAllignLayer = TFAllignLayer([name '_TFAllignLayer']);
lgraph = lgraph.addLayers(tfAllignLayer);

lgraph = lgraph.connectLayers(featureInput1.Name, [tfAllignLayer.Name '/in1']);
lgraph = lgraph.connectLayers(featureInput2.Name, [tfAllignLayer.Name '/in2']);

graspTFNet = dlnetwork(lgraph);

end