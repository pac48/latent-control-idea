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
allObjects = {'book', 'iphone_box'};
tmp = cellfun(@(x) ['nerf_' x], allObjects, 'UniformOutput', false);
nerf  = Nerf(tmp);

loftr = LoFTR();

%% init ros
rosshutdown()
rosinit('http://192.168.1.10:11311')
robot = Sawyer();
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');
jointCommandPub = rospublisher('/robot/limb/right/joint_command', 'DataFormat','struct');

%% create featureNet
featureNet = createFeatureNet(imageSize);

%% create TFNet
objects = {'book', 'iphone_box'};
TFNet = createTFNet(featureNet, objects);

%% create constNet
fov = 100;
constNet = createConstNet(objects, imageSize);
constNet = resetConstNet(constNet, objects);

%% load constNet
tmp = load('constNet.mat');
constNet = tmp.constNet;

%% create FullTFNet
fullTFNet = createFullTFNet(constNet, TFNet, ...
    { {'grab', 'iphone_box'}, ...
    {'place', 'book'} } );

%% get image from hand camera
close all
cam = webcam();

while 1
    img = cam.snapshot();
    image(img)
    initMsg = joint_sub.receive();
end

%% convert to dlarray
img = imresize(img, [720 1280]);
img = dlarray(double(img)./255, 'SSCB');

%% collect pick ipohone box demonstration
timeout = 10;
iphoneBoxMsgs = collectDemonstration(robot, joint_sub, timeout);

bodyNames = {'right_electric_gripper_base','right_gripper_l_finger_tip', 'right_gripper_r_finger_tip'};
interpLength = 1000;
[t, X, Xd] = convertMsgsToMatrix(robot, iphoneBoxMsgs, bodyNames, interpLength);
iphoneBoxData = {t, X, Xd}; 

%% playback ipohone box demonstration
close all
playBackDemonstration(robot, iphoneBoxMsgs)

%% collect place book demonstration
timeout = 10;
bookMsgs = collectDemonstration(robot, joint_sub, timeout);

bodyNames = {'right_electric_gripper_base','right_gripper_l_finger_tip', 'right_gripper_r_finger_tip'};
interpLength = 1000;
[t, X, Xd] = convertMsgsToMatrix(robot, bookMsgs, bodyNames, interpLength);
bookData = {t, X, Xd}; 

%% playback ipohone box demonstration
close all
playBackDemonstration(robot, bookMsgs)


%% train fullTFNet
fov = 69;
objectsPred = {'book', 'iphone_box'};

Z = featureNet.predict(img);
Trobot_cam2 = getTrobot_cam(robot, initMsg);
inputs{1} = containers.Map({'image_input', 'feature_input', 'Trobot_cam2_input'}, {img, Z, dlarray(reshape(Trobot_cam2, [], 1), 'CB')});
% targets{1} = containers.Map({'grab_iphone_box', 'place_book'}, {iphoneBoxData, bookData});
targets{1} = containers.Map();

clearCache(fullTFNet)
% for object = objectsPred
%     findInitMatch(fullTFNet, img, object{1})
% end

setDetectObjects(fullTFNet, objectsPred)
for i = 1:5
    setRotationDetection(fullTFNet, i<3)
    fullTFNet = trainFullTFNet(fullTFNet, inputs, targets, objectsPred, 5E-2);
end

%%
close all
setSkipNerf(fullTFNet, false);

[map, state] = getfullTFNetOutput(fullTFNet, inputs{1});

plotAllCorrespondence(fullTFNet, 1)
plotImageMSE(fullTFNet, img)

%% visualize point cloud
close all
Trobot_goal1 = map('PSMLayer_grab_iphone_box/Trobot_goal');
Trobot_goal2 = map('PSMLayer_place_book/Trobot_goal');

% while 1
    robot.setJointsMsg(initMsg);
    hold off
    robot.plotObject()
    plotTF(Trobot_cam2, '-')
    plotTF(Trobot_goal1, '-')
    plotTF(Trobot_goal2, '-')
    
    plotPointCloud(robot, fullTFNet, objects, {map}, Trobot_cam2)
    drawnow
% end

%% visualize demonstration with point cloud
figure
playBackDemonstration(robot, allMsg)
plotTF(Trobot_goal, '-')
plotPointCloud(robot, fullTFNet, objects, {map}, Trobot_cam2)

%% generate PSM trajecory and preview it
robot.setJointsMsg(joint_sub.receive());
robot.setJointsMsg(joint_sub.receive());
robot.setJointsMsg(joint_sub.receive());

% skillName = 'grab_iphone_box';
skillName = 'place_book';
[tPSM, XPSM, XdPSM] = generatePSMTraj(robot, bodyNames, map, skillName);

msgs = convertMatrixToMsgs(robot, bodyNames, joint_sub, tPSM, XPSM, XdPSM);
% [tTrack, XTrack, XdTrack] = convertMsgsToMatrix(robot, msgs, bodyNames, interpLength);
% plot(tTrack, XTrack)

close all

playBackDemonstration(robot, msgs)

%% execute PSM trajectory
gain = 2;
time_scale = 1;
is_reverse = false;
th = .3;


executeReference(gain, msgs, joint_sub, robot, jointCommandPub, [], time_scale, th, is_reverse)




%% get goal transforms
Trobot_phone = getObjectTransform(fullTFNet, map, 'iphone_box', Trobot_cam2);
Trobot_book = getObjectTransform(fullTFNet, map, 'book', Trobot_cam2);

close all
goalMsg = joint_sub.receive();
robot.setJointsMsg(goalMsg);
robot.plotObject()
Trobot_goal1 = robot.getBodyTransform('right_electric_gripper_base');
Trobot_goal2 = robot.getBodyTransform('right_gripper_l_finger_tip');
Trobot_goal3 = robot.getBodyTransform('right_gripper_r_finger_tip');
plotTF(Trobot_goal1, '-')
plotTF(Trobot_goal2, '-')
plotTF(Trobot_goal3, '-')

%%

target = cat(1, Trobot_goal1(1:3, end), Trobot_goal2(1:3, end), Trobot_goal3(1:3, end))

bodyNames = {'right_electric_gripper_base','right_gripper_l_finger_tip', 'right_gripper_r_finger_tip'}


commandMsg = rosmessage(jointCommandPub)

commandMsg.Names = robot.jointNames;
commandMsg.Mode = commandMsg.VELOCITYMODE;
commandMsg.Velocity = robot.getJoints()*0;

error = 1;
while norm(error) > .005
    robot.setJointsMsg(joint_sub.LatestMessage);

    T1 = robot.getBodyTransform('right_electric_gripper_base');
    T2 = robot.getBodyTransform('right_gripper_l_finger_tip');
    T3 = robot.getBodyTransform('right_gripper_r_finger_tip');
    curPoint = cat(1, T1(1:3, end),T2(1:3, end), T3(1:3, end))

    [JWrist, JFingerL, JFingerR] = robot.getControlPointsJacobians(bodyNames);
    J = cat(1, JWrist(1:3, :), JFingerL(1:3, :), JFingerR(1:3, :));

    error = target - curPoint;

    value = 1*error;
    qd = pinv(J)*value;

    if max(qd) > .2
        qd = .2*qd./max(qd);
    end

    commandMsg.Velocity = qd;
    jointCommandPub.send(commandMsg)

    % pause(1/50)
end





%% plot all transforms
% close all
% robot.plotObject
% for b = 1:length(robot.bodyNames)
%     Trobot_cam2 = robot.getBodyTransform(b);
%
% robot.setJointsMsg(joint_sub.receive());
%     plotTF(Trobot_cam2, '-')
%     drawnow
% end

%% show point cloud with robot

gripperBaseInd = 18;
gripperBaseInd = 24; % cam link
while 1
    msg = joint_sub.LatestMessage;
    if isempty(msg)
        continue
    end
    robot.setJointsMsg(msg);
    hold off
    robot.plotObject
    hold on

    T = robot.getBodyTransform(gripperBaseInd);

    point = T(1:3,end);
    line =  [point point+T(1:3,1)*.15];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','r')
    line =  [point point+T(1:3,2)*.15];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','g')
    line =  [point point+T(1:3,3)*.15];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','b')


    plotPointCloud(robot, fullTFNet, objects, {map}, Trobot_cam2)

    drawnow

end


%% convert to Msgs
msgs = convertMatrixToMsgs(robot, bodyNames, joint_sub, t, X, Xd);
close all
robot.setJointsMsg(joint_sub.receive());
playBackDemonstration(robot, msgs)


%%
gripperPub = rospublisher('/gripper_command')
msg = rosmessage(gripperPub)

msg.Data = false
send(gripperPub, msg)

msg.Data = true
send(gripperPub, msg)
