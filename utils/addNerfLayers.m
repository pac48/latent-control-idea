function lgraph = addNerfLayers(lgraph, nerf, objectCel, imageSize,...
    name_prefix, image_layer_name, ind_layer_name, global_tf_layer_name, feature_Trobot_cam2_layer_name)

allT = nerf.name2Frame(objectCel{1});

% allT   = allT(floor(linspace(1,length(allT), 20 )));

numTransforms = length(allT);
% knnLayer = fullyConnectedLayer((6)*numTransforms, 'Name', [name_prefix 'FCLayer']);
maxBatchSize = 100;
knnLayer = ConstLayer([name_prefix 'ConstLayer'], [6 numTransforms maxBatchSize]);
nerfLayer = NerfLayer([name_prefix 'NerfLayer'], allT, objectCel, imageSize(1), imageSize(2));
% prerender transforms

tfOffsetLayer = TFOffsetLayer([name_prefix 'TFOffsetLayer'], allT);
tfLayer = TFLayer([name_prefix 'TFLayer']);
[fl, fx, fy] = nerfLayer.getFValues();
projectionLayer = ProjectionLayer([name_prefix 'ProjectionLayer'], fl, fx,fy);

% psmLayer = PSMLayer(['grab_' nerfLayer.objectName], 9, 100, 1000);

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
% lgraph = lgraph.addLayers(psmLayer);
% lgraph = lgraph.addLayers(projectionLayer);

lgraph = lgraph.connectLayers(global_tf_layer_name, knnLayer.Name);

lgraph = lgraph.connectLayers(knnLayer.Name, [tfOffsetLayer.Name '/' tfOffsetLayer.InputNames{1}]);
% lgraph = lgraph.connectLayers(xyz_rpz_layer_name, [tfOffsetLayer.Name '/' tfOffsetLayer.InputNames{1}]);
% lgraph = lgraph.connectLayers(knnLayer.Name, [tfOffsetLayer.Name '/' tfOffsetLayer.InputNames{2}]);

lgraph = lgraph.connectLayers([tfOffsetLayer.Name '/' tfOffsetLayer.OutputNames{2}], [nerfLayer.Name '/' nerfLayer.InputNames{1}]);
lgraph = lgraph.connectLayers(image_layer_name, [nerfLayer.Name '/' nerfLayer.InputNames{2}]);

lgraph = lgraph.connectLayers(feature_Trobot_cam2_layer_name, [nerfLayer.Name '/' nerfLayer.InputNames{3}]);

lgraph = lgraph.connectLayers([nerfLayer.Name '/' nerfLayer.OutputNames{1}], [tfLayer.Name '/' tfLayer.InputNames{1}]);
lgraph = lgraph.connectLayers([tfOffsetLayer.Name '/' tfOffsetLayer.OutputNames{1}], [tfLayer.Name '/' tfLayer.InputNames{2}]);

% lgraph = lgraph.connectLayers([nerfLayer.Name '/' nerfLayer.OutputNames{7}], [psmLayer.Name]);

% lgraph = lgraph.connectLayers([tfLayer.Name '/' tfLayer.OutputNames{1}], projectionLayer.Name);

end

