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
global fov

w = 4032/4;
h = 3024/4;
% fov = 56; % needs to be same value as camera used for real images
% fov = 112; % iphone: needs to be same value as camera used for real images
% fov = 111;

% fov = 43; % realsense: needs to be same value as camera used for real images
fov = 87;
% fov = 69;
w = 1280;
h = 720;

imageSize = [h, w];


loftr = LoFTR();
% allObjects = {'background', 'book', 'iphone_box', 'plate', 'fork', 'blue_block'}; %, , 'plate', 'blue_block', 'fork'
allObjects = {'book', 'iphone_box'};
% objects = {'background'};

tmp = cellfun(@(x) ['nerf_' x], allObjects, 'UniformOutput', false);
nerf  = Nerf(tmp); %'nerf_blue_block',

%% create featureNet
featureNet = createFeatureNet(imageSize);

%% create TFNet
% objects = {'background'};
objects = {'book', 'iphone_box'};
TFNet = createTFNet(featureNet, objects);
% createPretrainData(featureNet, objects)
% [Zall, Tall] = TFProcessData(objects, 'pretrain');
% TFNet = trainTFNet(TFNet, Zall, Tall, 0.001);

%% create constNet
fov = 100;
constNet = createConstNet(objects, imageSize);
constNet = resetConstNet(constNet, objects);

%% create FullTFNet
fullTFNet = createFullTFNet(constNet, TFNet);

%% train
% fd = datastore('experiment_pick_place/data', 'ReadFcn', @(x) load(x).dataPoint, 'Type', 'file');
% allData = fd.readall();
% imgs = cellfun(@(x) double(imresize(x.img, imageSize))./255, allData, 'UniformOutput', false);
% imgs = cellfun(@(x) dlarray(double(x), 'SSCB'), imgs,  'UniformOutput', false);
% imgs = cat(4, imgs{:});


% fd = imageDatastore('calibration/');
fd = imageDatastore('data');
allData = fd.readall();
imgs = cellfun(@(x) double(imresize(x, imageSize))./255, allData, 'UniformOutput', false);
imgs = cellfun(@(x) dlarray(double(x), 'SSCB'), imgs,  'UniformOutput', false);
imgs = cat(4, imgs{:});


imgs = imgs(:, :, :, 1);
objectsPred = objects(1:end); 

% for object = objectsPred
%     findInitMatch(constNet, imgs, object{1})
% end
% constNet = resetConstNet(constNet, objects);

clearCache(constNet)
setDetectObjects(constNet, objectsPred)
% fov = 100;
% for i = 1:2
%     constNet = trainConstNet(constNet, imgs, objectsPred, 0.00003);
% end
fov = 69;
for i = 1:5
    constNet = trainConstNet(constNet, imgs, objectsPred, 0.00003);
end
% Tobj_cam_map = getObjectsTMap(constNet, imgs, objects);
% Z = featureNet.predict(imgs);
% TFNet = trainTFNet(TFNet, Z, Tobj_cam_map, 0.01);
% fullTFNet = createFullTFNet(constNet, TFNet);
% clearCache(fullTFNet)
% setDetectObjects(fullTFNet, objects)
% setSkipNerf(fullTFNet, false);

%% predict

maps = {};
for ind = 1:size(imgs, 4)
    %         curInd = ind;
    %         setSkipNerf(fullTFNet, false);
    %         [map, state] = getNetOutput(fullTFNet, imgs(:,:,:, ind), Z(:,ind));

    setSkipNerf(constNet, false);
    [map, state] = getNetOutput(constNet, imgs(:,:,:, ind), dlarray(ind,'CB'));

    maps = cat(2, maps, {map});
end
mapBatch = mergeMaps(maps);
plotAllCorrespondence(constNet, 1)
plotImageMSE(constNet, imgs)

%% ROS
rosshutdown()
rosinit('http://192.168.1.10:11311')
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');

%% create point cloud
robot = Sawyer();
% tmp = load('Trobot_background.mat');
% Trobot_background = tmp.Trobot_background;
msgForTestPose = joint_sub.LatestMessage;
Trobot_cam2 = getTrobot_cam(robot, msgForTestPose);

%%
close all



figure
plotPointCloud(robot, constNet, objects, maps(1), Trobot_cam2)
plotTF(Trobot_cam2, '-')

% p1 = Trobot_cam2(1:3,end);
% theta = 87/2;
% for s = [-1 1]
% vec = (Trobot_cam2(1:3, 1)*sind(theta) + s*Trobot_cam2(1:3, 2)*cosd(theta) )*3;
% p2 = p1 + vec;
% plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)])
% end

%% create Trobot_background

tmp = load('Mapsrobot_background.mat');
maps = tmp.maps;

Trobot_backgroundAll = [];

for ind = 1:3
    fileName = ['msgCamPos' num2str(ind) '.mat'];
    tmp = load(['calibration/' fileName]);
    robot.setJointsMsg(tmp.msg);

    map = maps{ind};
    Tcam1_background = extractdata(map('background_nerf_T_world_2_cam'));
    camLinkInd = 24; % cam link
    Trobot_cam2 = robot.getBodyTransform(camLinkInd);

    Tcam2_cam1 = eye(4);
    Tcam2_cam1(1:3, 1:3) = [0 0 -1
                            -1 0 0
                            0 1 0];

    Trobot_background = Trobot_cam2*Tcam2_cam1*Tcam1_background;
    Trobot_backgroundAll = cat(3, Trobot_backgroundAll, Trobot_background);

    plotPointCloud(robot, constNet, {'background'}, maps(ind), Trobot_background)

    point = Trobot_cam2(1:3,end);
    line =  [point point+Trobot_cam2(1:3,1)*1.9];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','r')
    line =  [point point+Trobot_cam2(1:3,2)*.15];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','g')
    line =  [point point+Trobot_cam2(1:3,3)*.15];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','b')

end


Rall = Trobot_backgroundAll(1:3, 1:3, :);
rpyAll = [];
for i = 1:size(Rall)
    rpyAll = cat(3, rpyAll, rotm2eul(Rall(:, :, i))');
end

Pall = Trobot_backgroundAll(1:3, end, :);
P = mean(Pall, 3);
rpy = mean(rpyAll, 3);
Trobot_background = [[eul2rotm(rpy') P]; [0 0 0 1]];

% save('Trobot_background', 'Trobot_background')

%% calculate transform from nerf
robot = Sawyer();

[fl, fx, fy] = getFValues(constNet);

out = cellfun(@(x) x.goal, allData, 'UniformOutput', false);
Tgoal = cat(3, out{:});
out = cellfun(@(x) x.start, allData, 'UniformOutput', false);
Tstart = cat(3, out{:});

robotBookPoints = Tgoal(1:3,end,:);
robotBoxPoints = Tstart(1:3,end,:);

robotBookPoints = robotBookPoints + Tgoal(1:3, 3,:)*0.13;
robotBoxPoints = robotBoxPoints + Tstart(1:3, 3,:)*0.13;

bookPoints = [];
boxPoints = [];

for i = 1:length(maps)
    [allBookPoints, ~] = getObjectPointCloud(maps{i}, 'book', fl, fx, fy);
    tmp = mean(allBookPoints,2);
    bookPoints = cat(3, bookPoints, tmp);

    [allBoxPoints, ~] = getObjectPointCloud(maps{i}, 'iphone_box', fl, fx, fy);
    tmp = mean(allBoxPoints, 2);
    boxPoints = cat(3, boxPoints, tmp);
end

robotBookPoints = double(reshape(robotBookPoints, 3, [])');
robotBoxPoints = double(reshape(robotBoxPoints, 3, [])');
bookPoints = double(reshape(bookPoints, 3, [])');
boxPoints = double(reshape(boxPoints, 3, [])');


% ptCloudRobot = pointCloud(cat(1,robotBookPoints, robotBoxPoints));
% ptCloudNerf = pointCloud(cat(1,bookPoints, boxPoints));
ptCloudRobot = pointCloud(cat(1, robotBookPoints));
ptCloudNerf = pointCloud(cat(1, bookPoints));

% find TF to bring nerf into robot frame
[tformEst, inlierIndex] = estgeotform3d(ptCloudNerf.Location, ptCloudRobot.Location, "rigid");
% [tformEst, inlierIndex] = estimateGeometricTransform3D(ptCloudRobot.Location, ptCloudNerf.Location, 'rigid');


figure

inliersPtCloudNerf = transformPointsForward(tformEst, ptCloudNerf.Location(inlierIndex,:) );
inliersPtCloudRobot = ptCloudRobot.Location(inlierIndex,:);

% inliersPtCloudNerf = ptCloudNerf.Location(inlierIndex,:);
% inliersPtCloudRobot = transformPointsForward(tformEst, ptCloudRobot.Location(inlierIndex,:));


scatter3(inliersPtCloudNerf(:,1), inliersPtCloudNerf(:,2), inliersPtCloudNerf(:,3))
hold on
scatter3(inliersPtCloudRobot(:,1), inliersPtCloudRobot(:,2), inliersPtCloudRobot(:,3))

T = eye(4);
T(1:3, 1:3) = tformEst.Rotation';
T(1:3, end) = tformEst.Translation;

%% PSMNET

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


