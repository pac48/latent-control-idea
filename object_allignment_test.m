close all
clear all

addpath('LoFTR/')
addpath('nerf/')
addpath('utils/')

loftr = LoFTR();
nerf = Nerf({'nerf_background', 'nerf_box', 'nerf_cup'});

w = 320*2;
h = 240*2;
fov = 70.8193; % needs to be same value as camera used for real images
imageSize = [h, w];
% [allT, allImgs] = nerf.name2Frame('nerf_background');
% testInd = 23;
% T0 =  allT{testInd};
% T0(1:3,end) = T0(1:3,end) + T0(1:3, 1);
% T0(1:3,end) = T0(1:3,end) + T0(1:3, 2);
% T0(1:3,end) = T0(1:3,end) + 1*T0(1:3, 3);

% T0(1:3, 1:3) = T0(1:3, 1:3)*eul2rotm([pi/12 0 0]);
% T0(1:3, 1:3) = T0(1:3, 1:3)*eul2rotm([0 pi/8 0]);
% T0(1:3, 1:3) = T0(1:3, 1:3)*eul2rotm([0 0 pi/12]);


% T0 = inv(T0); % world to cam

% T0(1:3,end) = T0(1:3,end) + [0;1;0];
% T0(1:3,end) = T0(1:3,end) + [1; 0; 0];


% file_name =  fullfile('instr/background2_data/images/', allNames{i}{1});
% imReal = allImgs{testInd};


%% nerf layer test

% rpyxyz = dlarray(zeros(6,1),'CB');
% img = dlarray(double(imReal), 'SSCB');
%
% layer0 = TFOffsetLayer(T0);
% layer1 = NerfLayer(nerf, loftr, {'nerf_background'}, imageSize(1), imageSize(2), fov);
% layer2 = TFLayer();
% layer3 = ProjectionLayer();
%
%
% Z = layer0.predict(rpyxyz);
% [nerf3dPoints, real2dPoints] = layer1.predict(Z, img);



%% init feature network
net = resnet18('Weights','imagenet');
inLayer = net.Layers(1);
featurePreLayers = [...
    %     imageInputLayer([imageSize(1:2) 3],'name','image_input', 'Normalization','zscore', 'Mean', ...
    %     inLayer.Mean,'StandardDeviation', inLayer.StandardDeviation);
    imageInputLayer([imageSize(1:2) 3],'name','image_input', 'Normalization','none');
    resize2dLayer('OutputSize', [224 224], 'name', 'resize', 'Method','bilinear')
    ];

featurePostLayers = [...
    functionLayer(@(x) dlarray(sin(2000*reshape(x, [], size(x,4))), 'CB'), 'Name','featurePostLayer', 'Formattable',true)
    ];


lgraph = net.layerGraph;
layers = lgraph.Layers;
for i = 20:length(layers)
    lgraph = lgraph.removeLayers(layers(i).Name);
end
lgraph = lgraph.removeLayers('data');
lgraph = lgraph.addLayers(featurePreLayers);
lgraph = lgraph.connectLayers('resize', 'conv1');
lgraph = lgraph.addLayers(featurePostLayers);
lgraph = lgraph.connectLayers(lgraph.Layers(end-3,:).Name, 'featurePostLayer');

featureNet = dlnetwork(lgraph);

%% init global transform network
% knnLayer = KNNLayer();
% for i = 1:size(allImgs)
% %     if i == testInd
% %         continue
% %     end
%     img = dlarray(double(allImgs{i})./255,'SSCB');
%     Z = gather(extractdata(featureNet.predict(img)));
%     knnLayer.addImage(Z, inv(allT{i}));
% end
%
% globalTransformPostLayers = [...
%     knnLayer()
%     ];
%
% lgraph = featureNet.layerGraph;
% lgraph = lgraph.addLayers(globalTransformPostLayers);
% lgraph = lgraph.connectLayers(lgraph.Layers(end-1,:).Name, 'KNNLayer');
%
% globalTransforNet = dlnetwork(lgraph);

% img = dlarray(double(imReal)./255,'SSCB');
% globalTransforNet.predict(img)
% allT{4}

%%
% lgraph = lgraphOld;
lgraph = featureNet.layerGraph;

backgroundConstLayers = ConstLayer('background_xyz_rpy', [6 1]);
lgraph = lgraph.addLayers(backgroundConstLayers);
lgraph = lgraph.connectLayers('image_input', backgroundConstLayers.Name);


name_prefix = 'background_nerf_';
image_layer_name = 'image_input';
feature_layer_name = 'featurePostLayer';
xyz_rpz_layer_name = backgroundConstLayers.Name;
lgraph = addNerfLayers(lgraph, featureNet, nerf, loftr, {'nerf_background'}, imageSize, fov, ...
    name_prefix, image_layer_name, feature_layer_name, xyz_rpz_layer_name);


cupConstLayers = ConstLayer('cup_xyz_rpy', [6 1]);
lgraph = lgraph.addLayers(cupConstLayers);
lgraph = lgraph.connectLayers('image_input', cupConstLayers.Name);
lgraph = addNerfLayers(lgraph, featureNet, nerf, loftr, {'nerf_cup'}, imageSize, fov, ...
    'cup_nerf_', image_layer_name, feature_layer_name, cupConstLayers.Name);

dlnet = dlnetwork(lgraph);

imReal = imread('test.jpg');
imReal = double(imresize(imReal, imageSize))./255;
img = dlarray(double(imReal), 'SSCB');
[nerfPoints2D, realPoints2D] = dlnet.predict(img)

% x = cat(1, nerfPoints2D(1,:), realPoints2D(1,:));
% y = cat(1, nerfPoints2D(2,:), realPoints2D(2,:));
% hold off
% plot(x,y)
% hold on
% plot(realPoints2D(1,:), realPoints2D(2,:),'LineStyle','none', 'Marker','.', 'MarkerSize', 8)

%% train
initialLearnRate = 2e-5;
decay = 0.0001;
momentum = 0.0;
velocity = [];
iteration = 0;

while 1
    tic
    iteration = iteration + 1;

    [loss, gradients, state] = dlfeval(@simpleModelGradients, dlnet, img);
    dlnet.State = state;
    loss
    % Determine learning rate for time-based decay learning rate schedule.
    learnRate = initialLearnRate/(1 + decay*iteration);

    % Update the network parameters using the SGDM optimizer.
    %     gradients(2,:).Value{1}
    %     gradients(1,:).Value{1}
    if ~isfinite(sum(gradients(1,:).Value{1}))
        why
    end
    [dlnet,velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
    toc
    %     pause(.2)
end



% %% network
%
% net = resnet18('Weights','imagenet');
% inLayer = net.Layers(1);
%
% preLayers = [
%     imageInputLayer([imageSize 3],'name','input', 'Normalization','zscore', 'Mean', ...
%     inLayer.Mean,'StandardDeviation', inLayer.StandardDeviation);
%     resize2dLayer('OutputSize', [224 224], 'name', 'resize', 'Method','bilinear')
%     ];
%
% nerfLayers = [
%     fullyConnectedLayer(6) % xyz rpy
%     TFOffsetLayer(T0)
%     NerfLayer(nerf, loftr, {'nerf_background'})
%     TFLayer()
%     ProjectionLayer()
%     ];
%
%
%
% lgraph = net.layerGraph;
% lgraph = lgraph.removeLayers('data');
% % lgraph = lgraph.removeLayers('fc1000');
% % lgraph = lgraph.removeLayers('prob');
% lgraph = lgraph.removeLayers('ClassificationLayer_predictions');
% lgraph = lgraph.addLayers(preLayers);
% lgraph = lgraph.connectLayers('resize', 'conv1');
% dlnet = dlnetwork(lgraph);
%
%
% % close all
% % figure(1)
% % tmp = net.predict(imresize(imReal, [224,224], 'bilinear'));
% % tmp = reshape(tmp,1,[]);
% % plot(tmp)
% % hold on
% % tmp =  dlnet.predict(dlarray(double(imReal),'SSCB'));
% % plot(tmp+.0001)
%
% %%
% [allT, allNames] = nerf.name2Frame('nerf_background');
% % T =  allT{testInd};
%
%
% % dlX = dlarray(Z,'CB');
%
%
% while 1
%     T(1:3,end) = T(1:3,end) + .1*T(1:3, 1);
%
%     tic
%     nerf.setTransform({'nerf_background', T})
%     imgNerf = uint8(255*nerf.renderObject(h, w, 'nerf_background'));
%
%     %     img1 = imresize(imgNerf, [240, 320]);
%     [mkptsReal, mkptsNerf, mconf] = loftr.predict(imReal, imgNerf);
%     toc
%     %     vec = mkpts0-mkpts1;
%
%
%
%     % figure(1)
%     % subplot(1,2,1)
%     % imshow(imReal)
%     % subplot(1,2,2)
%     % imshow(imgNerf)
%
%
%     figure(1)
%     subplot(1,2,1)
%     imshow(imReal)
%     subplot(1,2,2)
%     imshow(imgNerf)
%     drawnow
%     %
%     %     figure(2)
%     %     plotCorrespondence(imReal, imgNerf, mkptsReal, mkptsNerf, mconf)
% end

