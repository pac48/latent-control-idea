function lgraph = addNerfLayers(lgraph, featureNet, nerf, loftr, objectCel, imageSize, fov,...
    name_prefix, image_layer_name, global_tf_layer_name, xyz_rpz_layer_name)

[allT, allImgs] = nerf.name2Frame(objectCel{1});

numTransforms = length(allT);
knnLayer = fullyConnectedLayer((3+3)*numTransforms, 'Name', [name_prefix 'FCLayer']);
nerfLayer = NerfLayer([name_prefix 'NerfLayer'], nerf, loftr, objectCel, imageSize(1), imageSize(2), fov);
tfOffsetLayer = TFOffsetLayer([name_prefix 'TFOffsetLayer'], allT);
tfLayer = TFLayer([name_prefix 'TFLayer']);
projectionLayer = ProjectionLayer([name_prefix 'ProjectionLayer']);


% for i = 1:size(allImgs)
%     img = dlarray(double(allImgs{i})./255,'SSCB');
%     Z = gather(extractdata(featureNet.predict(img)));
%     knnLayer.addImage(Z, inv(allT{i}));
% end

% layers = [
%     TFOffsetLayer([name_prefix 'TFOffsetLayer'])
%     nerfLayer
%     TFLayer([name_prefix 'TFLayer'])
%     ProjectionLayer([name_prefix 'ProjectionLayer'])
%     ];

lgraph = lgraph.addLayers(knnLayer);
lgraph = lgraph.addLayers(tfOffsetLayer);
lgraph = lgraph.addLayers(nerfLayer);
lgraph = lgraph.addLayers(tfLayer);
lgraph = lgraph.addLayers(projectionLayer);

lgraph = lgraph.connectLayers(global_tf_layer_name, knnLayer.Name);

lgraph = lgraph.connectLayers(xyz_rpz_layer_name, [tfOffsetLayer.Name '/' tfOffsetLayer.InputNames{1}]);
% lgraph = lgraph.connectLayers(knnLayer.Name, [tfOffsetLayer.Name '/' tfOffsetLayer.InputNames{2}]);

lgraph = lgraph.connectLayers([tfOffsetLayer.Name '/' tfOffsetLayer.OutputNames{2}], [nerfLayer.Name '/' nerfLayer.InputNames{1}]);
lgraph = lgraph.connectLayers(image_layer_name, [nerfLayer.Name '/' nerfLayer.InputNames{2}]);

lgraph = lgraph.connectLayers([nerfLayer.Name '/' nerfLayer.OutputNames{1}], [tfLayer.Name '/' tfLayer.InputNames{1}]);
lgraph = lgraph.connectLayers([tfOffsetLayer.Name '/' tfOffsetLayer.OutputNames{1}], [tfLayer.Name '/' tfLayer.InputNames{2}]);

lgraph = lgraph.connectLayers(tfLayer.Name, projectionLayer.Name);

end

