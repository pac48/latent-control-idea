imds = imageDatastore('images/', 'ReadFcn', @readFunc);
% imds = imageDatastore('images_box/', 'ReadFcn', @readFunc);
% imds = imageDatastore('images_box_mask/', 'ReadFcn', @readFunc);
tmp = imds.readall();
imageSize = size(tmp{1});
allData = dlarray(zeros([imageSize length(tmp)]),'SSCB');
for i = 1:length(tmp)
    allData(:,:,:,i) = tmp{i}; 
end

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
out = gather(extractdata(dlnet.predict(allData)));
pred = reshape(out, [], length(tmp));

% num = length(gather(extractdata(reshape(out,[],1))));

% pred = zeros(length(allData), num);

% for i = 1:length(allData)
%     out = dlnet.predict(allData{i});
%     pred(i, :) = gather(extractdata(reshape(out,[],1)));
% end
% pred = pred./(mean(pred)+.1);

pred = [sin(2000*pred')];%.*sin(100*pred);

%% knn
close all

trainInds = 1:5:size(pred,1);
trainInds = cat(2,trainInds,size(pred,1));

Mdl = KDTreeSearcher(pred(trainInds, :),'Distance','cityblock');
    

for i = 1:size(pred, 1)
    if any(i==trainInds)
        continue
    end

    
    IdxNN = knnsearch(Mdl, pred(i, :),'K',1);
    
    IdxNN = trainInds(IdxNN);
    
    subplot(1,2,1)
    img = gather(extractdata(allData(:,:,:, i)));    
    imshow(img)
    subplot(1,2,2)
    img = gather(extractdata(allData(:,:,:, IdxNN)));    
    imshow(img)

    pause

end

