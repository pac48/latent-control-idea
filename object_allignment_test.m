
close all
clear all

addpath('LoFTR/')
addpath('SuperGlue/')
addpath('nerf/')
addpath('utils/')
addpath('utils/ransac2d/')

global loftr
loftr = LoFTR();
% loftr = SuperGlue();
allObjects = {'background', 'book', 'iphone_box', 'plate', 'fork', 'blue_block'}; %, , 'plate', 'blue_block', 'fork'

tmp = cellfun(@(x) ['nerf_' x], allObjects, 'UniformOutput', false);
global  nerf
nerf  = Nerf(tmp); %'nerf_blue_block',

global curInd

% numBackgroundTransforms = length(nerf.name2Frame('nerf_background'));
% numBoxTransforms = length(nerf.name2Frame('nerf_box'));
% numCupTransforms = length(nerf.name2Frame('nerf_cup'));

w = 4032/4;
h = 3024/4;
% 56 was pretty good
fov = 56;%57; %54.7505;%50.45;%54;%86.5797;%70.8193; % needs to be same value as camera used for real images
imageSize = [h, w];

buildNetword = false;


if buildNetword
    %% build
    net = resnet18('Weights','imagenet');
    inLayer = net.Layers(1);
    imageInput = imageInputLayer([imageSize(1:2) 3],'name','image_input', 'Normalization','none');
    featurePreLayers = [...
        %     imageInputLayer([imageSize(1:2) 3],'name','image_input', 'Normalization','zscore', 'Mean', ...
        %     inLayer.Mean,'StandardDeviation', inLayer.StandardDeviation);
        imageInput
        resize2dLayer('OutputSize', [224 224], 'name', 'resize', 'Method','bilinear')
        ];

    featurePostLayers = [...
        convolution2dLayer([1 1], 1, 'Stride', 1, 'Name','featurePostLayerIn')
        functionLayer(@(x) dlarray(reshape(x, [], size(x,4)), 'CB'), 'Name','featurePostLayer', 'Formattable',true)
%         functionLayer(@(x) dlarray(sin(2000*reshape(x, [], size(x,4))), 'CB'), 'Name','featurePostLayer', 'Formattable',true)
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
    lgraph = lgraph.connectLayers(lgraph.Layers(end-4,:).Name, 'featurePostLayerIn');

    featureNet = dlnetwork(lgraph);
    out = featureNet.predict(dlarray(zeros(10,10,3),'SSCB'));
    numFeatures = size(out,1);

    lgraph = layerGraph();
    featureInput = featureInputLayer(numFeatures, 'Name', 'feature_input');
    lgraph = lgraph.addLayers(featureInput);
    for object = allObjects
        name = object{1};
        TFLayerOut = fullyConnectedLayer(6, "Name", [name '_TF']);
        scaleLayer = functionLayer(@(x) .001*x, 'Name', [name 'scaleLayer']);
        lgraph = lgraph.addLayers(TFLayerOut);
        lgraph = lgraph.addLayers(scaleLayer);
        lgraph = lgraph.connectLayers(featureInput.Name, scaleLayer.Name);
        lgraph = lgraph.connectLayers(scaleLayer.Name, TFLayerOut.Name);
    end

    TFNet = dlnetwork(lgraph);

    lgraph = layerGraph();
    featureInput = featureInputLayer(1, 'Name', 'feature_input');
    lgraph = lgraph.addLayers(imageInput);
    lgraph = lgraph.addLayers(featureInput);
    % imageInputInd = featureInputLayer(1, 'Name', 'image_input_ind');
    % lgraph = lgraph.addLayers(imageInputInd);

    image_layer_name = imageInput.Name;
    ind_layer_name = '';%imageInputInd.Name;
    feature_layer_name = featureInput.Name;

    for object = allObjects
        name = object{1};
        lgraph = addNerfLayers(lgraph, featureNet, nerf, {['nerf_' name]}, imageSize, fov, ...
            [name '_nerf_'], image_layer_name, ind_layer_name, feature_layer_name);
    end

    dlnet = dlnetwork(lgraph);
    save('dlnetInit', 'dlnet', '-v7.3')
    save('featureNet', 'featureNet')
    save('TFNet', 'TFNet')
    %%
end

%% load init net
tmp = load('dlnetInit.mat');
dlnet = tmp.dlnet;
clearCache(dlnet)
tmp = load('featureNet.mat');
featureNet = tmp.featureNet;
tmp = load('TFNet.mat');
TFNet = tmp.TFNet;


%%
% imReal = imread('test1.jpg');
% imReal = imread('test.png');
% imReal = imread('test2.jpg');
% imReal = imread('IMG_0979.JPG');
% imReal = imread('IMG_0981.JPG');
% imReal = imread('IMG_0982.JPG');
% imReal = imread('IMG_1005.JPG');
% imReal = imread('IMG_1044.JPG');
% imReal = imread('IMG_1045.JPG');
% imReal = imread('IMG_1046.JPG');
% imReal = imread('IMG_1047.JPG');
% imReal = imread('data/6936B5EB-FEDD-4FFC-9437-EFE9BB846278.JPG');
% imReal = imread('0057.jpg');
% imReal = double(imresize(imReal, imageSize))./255;
% img = dlarray(double(imReal), 'SSCB');


imd = imageDatastore('data/');
imgs = imd.readall();
imgs = cellfun(@(x) double(imresize(x, imageSize))./255, imgs, 'UniformOutput', false);
imgs = cellfun(@(x) dlarray(double(x), 'SSCB'), imgs,  'UniformOutput', false);
imgs = cat(4, imgs{:});

tmp = load('data/iphone_box_grab.mat');
allMsg = tmp.allMsg;
robot = Sawyer();
gripperBaseInd = 18;
T = [];
for i = 1:length(allMsg)
    robot.setJointsMsg(allMsg(i))
    Ti = robot.getBodyTransform(gripperBaseInd);
    T = cat(3, T, Ti);
end
T = dlarray(T, 'SSB');


objects = {'background', 'book', 'iphone_box'}; % 'fork' 'blue_block',
for object = objects
    findInitMatch(dlnet, imgs, object{1})
end

Z = featureNet.predict(imgs);
setSkipNerf(dlnet, false);
for ind = 1:size(imgs, 4)
    [nerfPoints2D, realPoints2D] = dlnet.predict(imgs(:,:,:,ind), dlarray(ind,'CB'))
end

setSkipNerf(dlnet, true);
dlnet.predict(imgs(:,:,:,1), dlarray(1,'CB'));


%% train
numImages = size(imgs,4);

initialLearnRate = 5e-1;
decay = 0.005;
momentum = 0.95;
velocity = [];

iteration = 0;
index = 0;

averageGrad = [];
averageSqGrad = [];

while 1
    tic
    if mod(iteration, 100) == 0
        setSkipNerf(dlnet, false);
    else
        setSkipNerf(dlnet, true);
    end

    [loss, gradients, state, Tall] = dlfeval(@simpleModelGradients, dlnet, imgs, objects);
    loss

    learnRate = initialLearnRate/(1 + decay*iteration);
    [dlnet,velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);

    if mod(iteration, 100) == 0
        iteration
        index = mod(index + 1, numImages);
        subplot(1,1,1)
        plotAllCorrespondence(dlnet, index+1)
    end

    iteration = iteration + 1;
    toc

end

%% train
initialLearnRateTF = 1e-1;
decayTF = 0.000005;
momentumTF = 0.95;
velocityTF = [];

iteration = 0;

averageGrad = [];
averageSqGrad = [];

while 1
    tic

    [lossTF, gradients, state] = dlfeval(@TFModelGradients, TFNet, Z + .1*(rand(size(Z))-.5), Tall, objects, allObjects);
    lossTF
    learnRateTF = initialLearnRateTF/(1 + decayTF*iteration);
    [TFNet, velocityTF] = sgdmupdate(TFNet, gradients, velocityTF, learnRateTF, momentumTF);

    iteration = iteration + 1;
    toc

end


%% construct end to end TF net
lgraph = dlnet.layerGraph

feature_layer_name = 'feature_input';
lgraph = removeLayers(lgraph, feature_layer_name)
l = 1;
while l <= length(lgraph.Layers)
    layer = lgraph.Layers(l);
    if isa(layer, 'ConstLayer')
        lgraph = lgraph.removeLayers(layer.Name)
    end
    l = l+1;
end
l = 1;
while l <= length(lgraph.Layers)
    layer = lgraph.Layers(l);
    if isa(layer, 'TFOffsetLayer')
        lgraph = lgraph.removeLayers(layer.Name)
    end
    l = l+1;
end

for l = 1:length(TFNet.Layers)
    lgraph = lgraph.addLayers(TFNet.Layers(l))
end

for object = allObjects
    name = object{1};
    TFname = [name '_TF'];
    Scalename = [name 'scaleLayer']
    Nerfname = [name '_nerf_NerfLayer/in1'];
    TFLayername = [name '_nerf_TFLayer/in2'];
    lgraph = lgraph.connectLayers(feature_layer_name, Scalename);
    lgraph = lgraph.connectLayers(Scalename, TFname);
    lgraph = lgraph.connectLayers(TFname, Nerfname);
    lgraph = lgraph.connectLayers(TFname, TFLayername);
end
close all
plot(lgraph)

FullTFNet = dlnetwork(lgraph)
% save('FullTFNet', 'FullTFNet', '-v7.3')
%% test predict
% imd = imageDatastore('data_validation/');
imd = imageDatastore('data/');
imgs = imd.readall();
imgs = cellfun(@(x) double(imresize(x, imageSize))./255, imgs, 'UniformOutput', false);
imgs = cellfun(@(x) dlarray(double(x), 'SSCB'), imgs,  'UniformOutput', false);
imgs = cat(4, imgs{:});
Z = featureNet.predict(imgs);
numImages = size(imgs,4);

clearCache(FullTFNet)
% plot(Z)

for ind = 1:numImages
    curInd = ind;
    setSkipNerf(FullTFNet, false);
    [map, state] = getNetOutput(FullTFNet, imgs(:,:,:, ind), Z(:,ind));
end
setSkipNerf(FullTFNet, true);

for ind = 1:numImages
    figure
    subplot(1,3,1)
    image(extractdata(imgs(:,:,:,ind)))

    subplot(1,3,2)
    imgRender = plotRender(FullTFNet, ind);
    imgRender = double(imgRender)./255;

    subplot(1,3,3)
    tmpImg = extractdata(imgs(:,:,:,ind));
    %     base = .0001+sqrt(sum(tmpImg.^2, 3)).*sqrt(sum(imgRender.^2, 3));
    %     corr = sum(tmpImg.*imgRender, 3)./base;
    corr = 1-sum(abs(tmpImg - imgRender),3)./3;
    %     corr = corr.^8;
    corr = corr -.5;
    corr(corr < 0) = 0;
    corr = corr*(1/max(corr,[],"all"));

    imshow(corr)
    drawnow

    figure
    pause(1)
    plotAllCorrespondence(FullTFNet, ind)
    pause(1)
    
end

%%
%% train end to end
numImages = size(imgs,4);

initialLearnRate = 2e-4;
decay = 0.005;
momentum = 0.95;
velocity = [];

iteration = 0;
index = 0;

averageGrad = [];
averageSqGrad = [];

while 1
    tic
    if mod(iteration, 100) == 0
        setSkipNerf(FullTFNet, false);
    else
        setSkipNerf(FullTFNet, true);
    end

    [loss, gradients, state] = dlfeval(@FullTFModelGradients, FullTFNet, imgs, Z, objects); % need to change  simpleModelGradients
    loss
    %     dlnet.State = state;
    learnRate = initialLearnRate/(1 + decay*iteration);
    [FullTFNet,velocity] = sgdmupdate(FullTFNet, gradients, velocity, learnRate, momentum);

    if mod(iteration, 100) == 0
        iteration
        index = mod(index + 1, numImages);
        subplot(1,1,1);
        plotAllCorrespondence(FullTFNet, index+1)
    end

    iteration = iteration + 1;
    toc

    %     pause(.2)
end

%%
% for ind = 1:3
%     figure
%     subplot(1,3,1)
%     image(extractdata(imgs(:,:,:,ind)))
% 
%     subplot(1,3,2)
%     imgRender = plotRender(dlnet, ind);
%     imgRender = double(imgRender)./255;
% 
%     subplot(1,3,3)
%     tmpImg = extractdata(imgs(:,:,:,ind));
%     %     base = .0001+sqrt(sum(tmpImg.^2, 3)).*sqrt(sum(imgRender.^2, 3));
%     %     corr = sum(tmpImg.*imgRender, 3)./base;
%     corr = 1-sum(abs(tmpImg - imgRender),3)./3;
%     %     corr = corr.^8;
%     corr = corr -.5;
%     corr(corr < 0) = 0;
%     corr = corr*(1/max(corr,[],"all"));
% 
%     imshow(corr)
% 
% end
%%
figure
for ind = 1:size(imgs, 4)
     curInd = ind;
%     [map, state] = getNetOutput(FullTFNet, imgs(:,:,:, ind), dlarray(ind,'CB'));
     [map, state] = getNetOutput(FullTFNet, imgs(:,:,:, ind), Z(:,ind));
    
    pointsCam = 0.3011*extractdata(map('iphone_box_nerf_TFLayer/points_cam'));
%     pointsCam = 0.3074*extractdata(map('book_nerf_TFLayer/points_cam'));
    % pointsCam = extractdata(map('background_nerf_TFLayer/points_cam'));
    pointCam = mean(pointsCam, 2);

    rpyxyz = extractdata(TFNet.predict(Z(:, ind)));
    T = getT(rpyxyz(1:3), rpyxyz(4:6));

%     T = map('background_nerf_TFOffsetLayer/T'); % world to cam
    T = inv(T); % cam to world

    point = T(1:3,1:3)*pointCam + T(1:3, end);

    points = T(1:3,1:3)*pointsCam + T(1:3, end)
    hold on
    plot3(points(1, :), points(2, :), points(3, :), 'marker','.', 'MarkerSize',20)
    plot3(point(1, :), point(2, :), point(3, :), 'marker','.', 'MarkerSize',40)
%     plot3(pointsCam(1, :), pointsCam(2, :), pointsCam(3, :), 'marker','.', 'MarkerSize',20)
%     plot3(pointCam(1, :), pointCam(2, :), pointCam(3, :), 'marker','.', 'MarkerSize',40)
plot3(T(1, end), T(2, end), T(3, end), 'marker','.', 'MarkerSize',50)
end
% plot3(T(1, end), T(2, end), T(3, end), 'marker','.', 'MarkerSize',50)

% plot3(0, 0, 0, 'marker','.', 'MarkerSize',50)
axis equal


%% measure  scale
clc

p1 = [2.854,0.584,-0.127]; % objects
p2 = [2.344,0.584,-1.073];
l1 = norm(p1-p2)

p1 = [2.615,-0.055,-1.106]; % background
p2 = [2.325,-0.056,-1.278];
l2 = norm(p1-p2)

l2/l1

% width: real value 1.143
p1 = [0.580,-0.014,0.814];
p2 = [1.043,0.272,3.054];
l1 = norm(p1-p2)

p1 = [3.703,-0.106,2.267];
p2 = [3.372,-0.095,2.877];
l2 = norm(p1-p2)

l2/l1