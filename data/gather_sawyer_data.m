%% init ros
rosshutdown()
rosinit('http://192.168.1.10:11311')
robot = Sawyer();
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');

%% load constNet
load('constNet.mat')

%% get image from hand camera
close all
while 1
    cam = webcam();
    img = cam.snapshot();
    image(img)
end
%%
img = imresize(img, [720 1280]);
imwrite(img, 'data/0_0.jpg')
img = dlarray(double(img)./255, 'SSCB');

%% train constNet
% fov = 69;
% objectsPred = {'book', 'iphone_box'};
% 
% for object = objectsPred
%     findInitMatch(constNet, img, object{1})
% end
% constNet = resetConstNet(constNet, objects);
% % % clearCache(constNet)
% 
% setDetectObjects(constNet, objectsPred)
% for i = 1:5
%     constNet = trainConstNet(constNet, img, objectsPred, 0.00003);
% end

%% train fullTFNet
fov = 69;
objectsPred = {'book', 'iphone_box'};

Z = featureNet.predict(img);

clearCache(fullTFNet)
% for object = objectsPred
%     findInitMatch(fullTFNet, img, object{1})
% end


setDetectObjects(fullTFNet, objectsPred)
for i = 1:5
    fullTFNet = trainFullTFNet(fullTFNet, img, Z, objectsPred, 5E-2);
end

%%
close all
setSkipNerf(fullTFNet, false);

% [map, state] = getNetOutput(constNet, img, dlarray(1,'CB'));
[map, state] = getfullTFNetOutput(fullTFNet, img, Z);

plotAllCorrespondence(fullTFNet, 1)
plotImageMSE(fullTFNet, img)

%%
close all
Trobot_cam2 = getTrobot_cam(robot, joint_sub.LatestMessage);

% while 1
    robot.setJointsMsg(joint_sub.receive());
    plotPointCloud(robot, fullTFNet, objects, {map}, Trobot_cam2)
    plotTF(Trobot_cam2, '-')
    drawnow
% end


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

%% joint reading
% tmp = load("calibrate/T.mat");
% Toffset = tmp.T;
%
% robot.bodyNames{realSenseInd}
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


    %     T = robot.getBodyTransform(realSenseInd);
    %     point = T(1:3,end);
    %     line =  [point point+T(1:3,1)*.05];
    %     hold on
    %     plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','r')
    %     line =  [point point+T(1:3,2)*.05];
    %     hold on
    %     plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','g')
    %     line =  [point point+T(1:3,3)*.05];
    %     hold on
    %     plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','b')

    drawnow

end
%% build message from manual movment
allMsg = [];
allMsg = cat(1, allMsg, joint_sub.LatestMessage)

%% manual movement
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');

allMsg = [];


robot.setJointsMsg(receive(joint_sub))
q = robot.getJoints();

count = 0;
while 1
    jointMsg = joint_sub.LatestMessage;
    robot.setJointsMsg(jointMsg);
    if (norm(q-robot.getJoints()) < .02 )
        continue
    end
    q = robot.getJoints();
    allMsg = cat(1, allMsg, joint_sub.LatestMessage)
    count = count + 1
end

%% playback

gripperBaseInd = 18;
for i = 1:length(allMsg)
    msg = allMsg(i);
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

    drawnow
    pause(.1)

end

%%
gripperPub = rospublisher('/gripper_command')
msg = rosmessage(gripperPub)

msg.Data = false
send(gripperPub, msg)

msg.Data = true
send(gripperPub, msg)
