function featureNet = createFeatureNet(imageSize)

net = resnet18('Weights','imagenet');
imageInput = imageInputLayer([imageSize(1:2) 3],'name','image_input', 'Normalization','none');
featurePreLayers = [...
    imageInput
    resize2dLayer('OutputSize', [224 224], 'name', 'resize', 'Method','bilinear')
    ];

lgraph = net.layerGraph;
layers = lgraph.Layers;
for i = 70:length(layers) %20:length(layers)
    lgraph = lgraph.removeLayers(layers(i).Name);
end
lgraph = lgraph.removeLayers('data');
lgraph = lgraph.addLayers(featurePreLayers);
lgraph = lgraph.connectLayers('resize', 'conv1');

featureNet = dlnetwork(lgraph);

end