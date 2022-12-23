[allTCell, allImgsCell] = tp.name2Frame('nerf_background');
imageSize = [240, 320, 3];
allImgs = dlarray(zeros([imageSize length(allTCell)]),'SSCB');
allT  = dlarray(zeros([4 4 length(allTCell)]),'SSB');

for i = 1:length(allImgsCell)
    img = allImgsCell{i};
    img = imresize(img, imageSize(1:2));
    img = double(img)./255;
    allImgs(:,:,:,i) = img;
    allT(:,:,i) = allTCell{i};
end
%%

net = resnet18('Weights','imagenet');
inLayer = net.Layers(1);


preLayers = [
    imageInputLayer([imageSize(1:2) 3],'name','input', 'Normalization','zscore', 'Mean', ...
    inLayer.Mean,'StandardDeviation', inLayer.StandardDeviation);
    resize2dLayer('OutputSize', [224 224], 'name', 'resize', 'Method','bilinear')
    ];


postLayers = [
    fullyConnectedLayer(6)
];


lgraphPre = net.layerGraph;
layers = lgraphPre.Layers;

% for i = 13:length(layers)
for i = 20:length(layers)
% for i = 68:length(layers)
    lgraphPre = lgraphPre.removeLayers(layers(i).Name);
end
netOutLayerName = lgraphPre.Layers(end).Name;

lgraphPre = lgraphPre.removeLayers('data');
lgraphPre = lgraphPre.addLayers(preLayers);
lgraphPre = lgraphPre.connectLayers('resize', 'conv1');
dlnetPre = dlnetwork(lgraphPre);

features = dlnetPre.predict(allImgs);
features = reshape(features,[], size(features, 4));
features = dlarray(features, 'CB');


lgraph = layerGraph([ ...
    featureInputLayer(size(features,1), 'name', 'input')
    postLayers
    ]);
% lgraph = lgraph.addLayers(postLayers);
% lgraph = lgraph.addLayers(featureInputLayer(100,'name','input'));
dlnet = dlnetwork(lgraph);

%% freeze learnable parameters
% factor = 0;
% learnables = learnables(1:end-2,:);
% 
% numLearnables = size(learnables,1);
% for i = 1:numLearnables
%     layerName = learnables.Layer(i);
%     parameterName = learnables.Parameter(i);
%     
%     dlnet = setLearnRateFactor(dlnet, layerName, parameterName, factor);
% end

%% train
initialLearnRate = 5e-8;
decay = 0.0001;
momentum = 0.98;
velocity = [];
iteration = 0;

while 1
    iteration = iteration + 1;
%     ind = mod(iteration:iteration+10, size(allT, 3))+1;
%     [loss, gradients, state] = dlfeval(@modelGradients, dlnet, allImgs(:,:,:,ind), allT(:,:,ind));
  [loss, gradients, state] = dlfeval(@modelGradients, dlnet, features, allT);
    dlnet.State = state;
    loss
    learnRate = initialLearnRate/(1 + decay*iteration);
    [dlnet,velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
end
save('dlnet', 'dlnet')

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

