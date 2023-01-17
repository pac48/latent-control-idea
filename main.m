%% init globals and oaths 
close all
clear all

addpath('LoFTR/')
addpath('SuperGlue/')
addpath('nerf/')
addpath('utils/')
addpath('utils/ransac2d/')

addpath('TFNet/')
addpath('ConstNet/')
addpath('FullTFNet/')
addpath('FeatureNet/')
addpath('PSMNet/')


global loftr
global nerf
global curInd

loftr = LoFTR();
allObjects = {'background', 'book', 'iphone_box', 'plate', 'fork', 'blue_block'}; %, , 'plate', 'blue_block', 'fork'
objects = {'background', 'book', 'iphone_box'};

tmp = cellfun(@(x) ['nerf_' x], allObjects, 'UniformOutput', false);
nerf  = Nerf(tmp); %'nerf_blue_block',

w = 4032/4;
h = 3024/4;
fov = 56; % needs to be same value as camera used for real images
imageSize = [h, w];

%% create featureNet
featureNet = createFeatureNet(imageSize);

%% create TFNet
TFNet = createTFNet(featureNet, objects);
% createPretrainData(featureNet, objects)
% [Zall, Tall] = TFProcessData(objects, 'pretrain');
% TFNet = trainTFNet(TFNet, Zall, Tall, 0.001);

%% create constNet
constNet = createConstNet(objects, imageSize, fov);
constNet = resetConstNet(constNet, objects);

%% create FullTFNet
fullTFNet = createFullTFNet(constNet, TFNet);

%% predict-train
% imd = imageDatastore('experiment_pick_place/img/');
% imgs = imd.readall();
fd = datastore('experiment_pick_place/data', 'ReadFcn', @(x) load(x).dataPoint, 'Type', 'file');
allData = fd.readall();
imgs = cellfun(@(x) double(imresize(x.img, imageSize))./255, allData, 'UniformOutput', false);
imgs = cellfun(@(x) dlarray(double(x), 'SSCB'), imgs,  'UniformOutput', false);
imgs = cat(4, imgs{:});

% imgs = imgs(:,:,:,1:3);

constNet = resetConstNet(constNet, objects);
[Tobj_cam_map, constNet] = trainConstNet(constNet, imgs, objects, 0.01);

Z = featureNet.predict(imgs);
% TFNet = trainTFNet(TFNet, Z, Tobj_cam_map, 0.01);
% fullTFNet = createFullTFNet(constNet, TFNet);
% clearCache(fullTFNet)
% setDetectObjects(fullTFNet, objects)
% setSkipNerf(fullTFNet, false);

maps = {};
for ind = 1:size(imgs, 4)
    curInd = ind;
%     setSkipNerf(fullTFNet, false);
%     [map, state] = getNetOutput(fullTFNet, imgs(:,:,:, ind), Z(:,ind));
        setSkipNerf(constNet, false);
     [map, state] = getNetOutput(constNet, imgs(:,:,:, ind), dlarray(ind,'CB'));
    maps = cat(2, maps, {map});
end
mapBatch = mergeMaps(maps);

% plotAllCorrespondence(fullTFNet, 1)
% plotImageMSE(fullTFNet, imgs)

plotAllCorrespondence(constNet, 1)
plotImageMSE(constNet, imgs)

%% create point cloud
robot = Sawyer();
plotPointCloud(robot, map, 1)



%% PSMNET
robot = Sawyer();

out = cellfun(@(x) x.goal, allData, 'UniformOutput', false);
Tgoal = cat(3, out{:});

outInv = cellfun(@(x) inv(x.goal), allData, 'UniformOutput', false);
TgoalInv = cat(3, outInv{:});

object = objects{2};
graspTFNet = createGraspTFNet(object);

%test synthetic
% tmp = reshape(Tobj_cam, 6 , 1, []);
% TTobj_cam = getT(tmp(1:3, 1, :), tmp(4:6, 1, :));
% offsetT = [[eul2rotm([pi/2 0 0]) [.1; .2; .3]];[0 0 0 1]];
% Trobot_grasp = pagemtimes(offsetT, TTobj_cam); 

key = [object '_nerf_T_world_2_cam'];
out = mapBatch(key);
% Tcam_obj = cat(2, out{:});
% Tcam_obj = reshape(Tcam_obj, 6 , 1, []);
% Tcam_obj = getT(Tcam_obj(1:3, 1, :), Tcam_obj(4:6, 1, :));
% Tcam_obj = extractdata(Tcam_obj);

Tcam_obj = cat(3, out{:});

Tcam_obj = dlarray(reshape(Tcam_obj, 16, []), 'CB'); 


key = 'background_nerf_T_world_2_cam';
out = mapBatch(key);
% Tcam_background = cat(2, out{:});
% Tcam_background = reshape(Tcam_background, 6 , 1, []);
% Tcam_background = getT(Tcam_background(1:3, 1, :), Tcam_background(4:6, 1, :));

Tcam_background = cat(3, out{:});


% Trobot_grasp =  (Trobot_background)*((scale)Tbackground_cam)*((scale)*Tcam_obj)*(Tobj_grasp)

Tbackground_cam = extractdata(reshape(Tcam_background, 4, 4, []));
Tobj_cam = extractdata(reshape(Tcam_obj, 4, 4, []));
for i =1:size(Tbackground_cam, 3)
    Tbackground_cam(:,:,i) = inv(Tbackground_cam(:,:,i));
    Tobj_cam(:,:,i) = inv(Tobj_cam(:,:,i));
end
Tbackground_cam = dlarray(reshape(Tbackground_cam, 16, []), 'CB');
Tobj_cam = dlarray(reshape(Tobj_cam, 16, []), 'CB');



Trobot_grasp = dlarray(reshape(Tgoal, 4, 4, []), 'SSB'); 
Tgrasp_robot = dlarray(reshape(TgoalInv, 4, 4, []), 'SSB'); 
graspTFNet = trainGraspTFNet(graspTFNet, Tbackground_cam, Tcam_obj, Trobot_grasp, Tgrasp_robot, 1E-2);

%% pred
[Trobot_graspPred, Trobot_camPred, Tcam_graspPred, Tgrasp_robotPred] = graspTFNet.predict(Tbackground_cam, Tcam_obj)
figure

robot.plotObject()
for i = 1:size(Trobot_grasp,3)
    plotTF(Trobot_grasp(:,:,i), '--')
    plotTF(Trobot_graspPred(:,:,i), '-')
    Trobot_camPred_i = Trobot_camPred(:,:,i);
    Trobot_camPred_i(1:3,end) = Trobot_camPred_i(1:3,end)./Trobot_camPred_i(end, end);
    plotTF(Trobot_camPred_i, '-')
%  plotTF(Tcam_graspPred(:,:,i), '-')
   
end
axis equal



% psmNet = trainPSMNet(psmNet, Tobj_cam, Trobot_grasp, objects)

% T_to_from

% have from TFNet: Tobj_cam
% have from demonstration: Trobot_grasp
% I want Tgrasp_cam
% Trobot_grasp*(Tgrasp_cam) = Trobot_cam
% (Trobot_obj)*Tobj_cam = Trobot_cam





% Tgrasp_obj: fixed 
% Tbackground_robot: fixed

% Tobj_robot: variable
% Tobj_cam: variable
% Tbackground_cam: variable

% Tgrasp_robot =  (Tgrasp_obj)*Tobj_cam*Tcam_background*(Tbackground_robot)


% Tgrasp_obj*Tobj_robot = Tgrasp_robot
% Tobj_robot = Tobj_cam*Tcam_robot


