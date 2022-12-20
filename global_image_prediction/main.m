imds = imageDatastore('images/', 'ReadFcn', @readFunc);
% imds = imageDatastore('images_box/', 'ReadFcn', @readFunc);
% imds = imageDatastore('images_box_mask/', 'ReadFcn', @readFunc);
allData = imds.readall();

imageSize = size(allData{1});

net = resnet18('Weights','imagenet');
inLayer = net.Layers(1);



preLayers = [
    imageInputLayer([imageSize(1:2) 3],'name','input', 'Normalization','zscore', 'Mean', ...
    inLayer.Mean,'StandardDeviation', inLayer.StandardDeviation);
    resize2dLayer('OutputSize', [224 224], 'name', 'resize', 'Method','bilinear')
    ];




lgraph = net.layerGraph;
layers = lgraph.Layers;

% for i = 13:length(layers)
for i = 20:length(layers)
% for i = 68:length(layers)
    lgraph = lgraph.removeLayers(layers(i).Name);
end


lgraph = lgraph.removeLayers('data');
% lgraph = lgraph.removeLayers('ClassificationLayer_predictions');
% lgraph = lgraph.removeLayers('prob');
% lgraph = lgraph.removeLayers('pool5');
% lgraph = lgraph.removeLayers('fc1000');
lgraph = lgraph.addLayers(preLayers);
lgraph = lgraph.connectLayers('resize', 'conv1');
dlnet = dlnetwork(lgraph);


%%
out = dlnet.predict(allData{1});
num = length(gather(extractdata(reshape(out,[],1))));

pred = zeros(length(allData), num);

for i = 1:length(allData)
    out = dlnet.predict(allData{i});
    pred(i, :) = gather(extractdata(reshape(out,[],1)));
end
pred = pred./(mean(pred)+.1);

pred = sin(1000*pred);%.*sin(100*pred);

%%
trainInds = 1:5:size(pred,1);
Mdl = KDTreeSearcher(pred(trainInds, :),'Distance','cityblock');
    

for i = 1:size(pred,1)
    if any(i==trainInds)
        continue
    end

    
    IdxNN = knnsearch(Mdl, pred(i, :),'K',1);
    
    IdxNN = trainInds(IdxNN);
    
    subplot(1,2,1)
    img = allData{i};    
    img = extractdata(img);
    imshow(img)
    subplot(1,2,2)
    img = allData{IdxNN};    
    img = extractdata(img);
    imshow(img)

    pause

end

