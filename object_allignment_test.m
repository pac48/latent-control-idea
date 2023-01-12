
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
allObjects = {'background', 'book', 'iphone_box', 'plate', 'blue_block'}; %,

tmp = cellfun(@(x) ['nerf_' x], allObjects, 'UniformOutput', false);
global  nerf
nerf  = Nerf(tmp); %'nerf_blue_block',

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
        convolution2dLayer([3 3], 1, 'Stride', 3, 'Name','featurePostLayerIn')
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
    lgraph = lgraph.connectLayers(lgraph.Layers(end-4,:).Name, 'featurePostLayerIn');

    featureNet = dlnetwork(lgraph);
    out = featureNet.predict(dlarray(zeros(10,10,3),'SSCB'));
    numFeatures = size(out,1);

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
    %%
end

%% load init net
tmp = load('dlnetInit.mat');
dlnet = tmp.dlnet;
clearCache(dlnet)

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


objects = {'background', 'book', 'iphone_box'}; % 'blue_block', 
for object = objects
    findInitMatch(dlnet, imgs, object{1})
end


Z = featureNet.predict(imgs);
setSkipNerf(dlnet, false);
[nerfPoints2D, realPoints2D] = dlnet.predict(imgs(:,:,:,1), dlarray(1,'CB'))
setSkipNerf(dlnet, true);
 dlnet.predict(imgs(:,:,:,1), dlarray(1,'CB'));


% x = cat(1, nerfPoints2D(1,:), realPoints2D(1,:));
% y = cat(1, nerfPoints2D(2,:), realPoints2D(2,:));
% hold off

% plot(x,y)
% hold on
% plot(realPoints2D(1,:), realPoints2D(2,:),'LineStyle','none', 'Marker','.', 'MarkerSize', 8)

% dlnetOld = dlnet;

%% train
% clearCache(dlnet)
% dlnet = dlnetOld;

initialLearnRate = 5e-1;
decay = 0.0001;
momentum = 0.95;
velocity = [];
iteration = 0;


averageGrad = [];
averageSqGrad = [];

while 1
    tic
    if mod(iteration, 100) == 0
        setSkipNerf(dlnet, false);
    else
        setSkipNerf(dlnet, true);
    end


    [loss, gradients, state] = dlfeval(@simpleModelGradients, dlnet, imgs, allObjects);
    dlnet.State = state;
    loss
    % Determine learning rate for time-based decay learning rate schedule.
    learnRate = initialLearnRate/(1 + decay*iteration);
    [dlnet,velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
    %     [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration+1);%, learnRate, .99, 0.95

    if mod(iteration, 100) == 0
        iteration
        plotAllCorrespondence(dlnet, 3)
    end
    iteration = iteration + 1;
    toc

    %     pause(.2)
end
%%
for ind = 1:3
figure
subplot(1,3,1)
image(extractdata(imgs(:,:,:,ind)))

subplot(1,3,2)
imgRender = plotRender(dlnet, ind);
imgRender = double(imgRender)./255-.5;

subplot(1,3,3)
tmpImg = extractdata(imgs(:,:,:,ind))-.5;
base = .0001+sqrt(sum(tmpImg.^2, 3)).*sqrt(sum(imgRender.^2, 3));
corr = sum(tmpImg.*imgRender, 3)./base;
imshow(.999*corr)

end