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

fov = 69;
w = 1280;
h = 720;
imageSize = [h, w];
% allObjects = {'book'};
allObjects = {'drawer'};
% allObjects = {'iphone_box'};
tmp = cellfun(@(x) ['nerf_' x], allObjects, 'UniformOutput', false);
nerf  = Nerf(tmp);

loftr = LoFTR();

%% init ros
rosshutdown()
rosinit('http://192.168.1.10:11311')
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');
jointCommandPub = rospublisher('/robot/limb/right/joint_command', 'DataFormat','struct');

%%
robot = Sawyer();

% create featureNet
featureNet = createFeatureNet(imageSize);

% create TFNet
objects = allObjects;
TFNet = createTFNet(featureNet, objects);

% create constNet
fov = 100;
constNet = createConstNet(objects, imageSize);
constNet = resetConstNet(constNet, objects);


%% create FullTFNet
% task = {'open', 'drawer'};
task = {'pull', 'drawer'};
% task = {'pick', 'iphone_box'};
% task = {'place', 'book'};

fullTFNet = createFullTFNet(constNet, TFNet, ...
    { task} ...
    );

%% load demonstration
demoNum = 3;
% postfix = 'clutter';
postfix = 'lighting';
% postfix = '';

tmp = load(['data/' task{1} task{2} postfix num2str(demoNum) '.mat']);
img = tmp.datapoint.img;
img = imresize(img, [720 1280]);
img = dlarray(img, 'SSCB');
initMsg = tmp.datapoint.msg(1);
initMsgOriginal = tmp.datapoint.msg(1);
msgs = tmp.datapoint.msg;

skillName = [task{1} '_' task{2}];
bodyNames = {'right_electric_gripper_base','right_gripper_l_finger_tip', 'right_gripper_r_finger_tip'};
interpLength = 1000;
[t, X, Xd] = convertMsgsToMatrix(robot, msgs, bodyNames, interpLength);
data = {t, X, Xd};

if strcmp(task{1}, 'place') || strcmp(task{1}, 'pull')
    if strcmp(task{1}, 'place')
        name = ['pickiphone_box' postfix];
    elseif strcmp(task{1}, 'pull')
        name = ['opendrawer' postfix];
    else
        assert(0)
    end

    tmp = load(['data/' name num2str(demoNum) '.mat']);
    img = tmp.datapoint.img;
    img = imresize(img, [720 1280]);
    img = dlarray(img, 'SSCB');
    initMsg = tmp.datapoint.msg(1);

end

%% playback drawer box demonstration
close all
playBackDemonstration(robot, msgs)


%% get image from hand camera
close all
cam = webcam();

while 1
    img = cam.snapshot();
    initMsg = joint_sub.receive();
    image(img)
end
%% convert to dlarray
img = imresize(img, [720 1280]);
img = dlarray(double(img)./255, 'SSCB');

%% train fullTFNet
% fov = 69;
% fov = 80% good for drawer
% fov = 92% +2
fov = 86.25;

% objectsPred = {'book', 'iphone_box'};
objectsPred = allObjects;%(2:end);

Z = featureNet.predict(img);
Trobot_cam2 = getTrobot_cam(robot, initMsg);

inputs{1} = containers.Map({'image_input', 'feature_input', 'Trobot_cam2_input'}, {img, Z, dlarray(reshape(Trobot_cam2, [], 1), 'CB')});

targets{1} = containers.Map();
% targets{1} = containers.Map({skillName}, {data});

clearCache(fullTFNet)
for object = objectsPred
    findInitMatch(fullTFNet, img, object{1})
end

setDetectObjects(fullTFNet, objectsPred)
for i = 1:50
    if i < 10
        fov = 120;
    else
        fov =  86.25;
    end
    setRotationDetection(fullTFNet, i<=1)
    setTranslationDetection(fullTFNet, i<=10)
    fullTFNet = trainFullTFNet(fullTFNet, inputs, targets, objectsPred, 5E-2);
end

%%
close all
setRotationDetection(fullTFNet, false)
setTranslationDetection(fullTFNet, false)
setSkipNerf(fullTFNet, false);

[map, state] = getfullTFNetOutput(fullTFNet, inputs{1});

plotAllCorrespondence(fullTFNet, 1)
plotImageMSE(fullTFNet, img)

%% visualize point cloud
close all
% Trobot_goal1 = map('PSMLayer_grab_iphone_box/Trobot_goal');
% % Trobot_goal2 = map('PSMLayer_place_book/Trobot_goal');

Trobot_goal1 = map(['PSMLayer_' skillName '/Trobot_goal']);


% while 1
robot.setJointsMsg(initMsg);
% robot.setJointsMsg(joint_sub.receive());
hold off

robot.plotObject()
plotPointCloud(robot, fullTFNet, objects, {map}, Trobot_cam2)

%     plotTF(Trobot_cam2, '-')
plotTF(Trobot_goal1, '-')
%     plotTF(Trobot_goal2, '-')


drawnow
% end


%% generate PSM trajecory and preview it
% robot.setJointsMsg(joint_sub.receive());
% robot.setJointsMsg(joint_sub.receive());

% startMsg = joint_sub.receive();
% startMsg = initMsg;
startMsg = initMsgOriginal;

robot.setJointsMsg(startMsg);

[tPSM, XPSM, XdPSM] = generatePSMTraj(robot, bodyNames, map, skillName);

msgs = convertMatrixToMsgs(robot, bodyNames, startMsg, tPSM, XPSM, XdPSM);
% [tTrack, XTrack, XdTrack] = convertMsgsToMatrix(robot, msgs, bodyNames, interpLength);
% plot(tTrack, XTrack)

% msgs = convertMatrixToMsgs(robot, bodyNames, joint_sub, t, X, Xd);

close all

playBackDemonstration(robot, msgs)

tmp = XPSM;
% tmp(4:6,:) = tmp(4:6,:) + tmp(1:3,:);
% tmp(7:9,:) = tmp(7:9,:)+ tmp(1:3,:);
plot3(tmp([1 4 7],:)', tmp([2 5 8],:)', tmp([3 6 9],:))

robot.plotObject()
plotPointCloud(robot, fullTFNet, objects, {map}, Trobot_cam2)
plotTF(Trobot_goal1, '-')


%% execute PSM trajectory
gain = 2;
time_scale = 3;
is_reverse = false;
th = .3;

executeReference(gain, msgs, joint_sub, robot, jointCommandPub, [], time_scale, th, is_reverse)

%% execute save 1
gain = 2;
time_scale = 2;
is_reverse = false;
th = .3;

tmp = load(['data/' task{1} task{2} postfix num2str(demoNum) '.mat']);
datapoint = tmp.datapoint;
executeReference(gain, datapoint.msg, joint_sub, robot, jointCommandPub, [], time_scale, th, true)

robot.setJointsMsg(datapoint.msg(1));
[tPSM, XPSM, XdPSM] = generatePSMTraj(robot, bodyNames, map, skillName);

msgs = convertMatrixToMsgs(robot, bodyNames, datapoint.msg(1), tPSM, XPSM, XdPSM);
%% execute save 2
pause(2)

executeReference(gain, msgs, joint_sub, robot, jointCommandPub, [], time_scale, th, is_reverse)

resultPSM = struct('t',tPSM, 'X',XPSM, 'Xd',XdPSM, 'msg', msgs)
fileName = ['traj_results/psm_trajs_' task{1} task{2} '/' num2str(demoNum) postfix '.mat'];
save(fileName,"resultPSM")


%%
pub = rospublisher('/io/end_effector/right_gripper/command');
msgGrpper = rosmessage(pub);
%% close
msgGrpper.Op = "set";
msgGrpper.Args = '{"signals": {"position_m": {"data": [0.0], "format": {"type": "float"}}}}';
msgGrpper.Time = rostime('now');

pub.send(msgGrpper)

%% open
msgGrpper.Op = "set";
msgGrpper.Args = '{"signals": {"position_m": {"data": [0.041667], "format": {"type": "float"}}}}';
msgGrpper.Time = rostime('now');
pub.send(msgGrpper)

