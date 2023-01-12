%% config cam
% m = mobiledev;
% cam = camera(m,'back');
% cam.Resolution = '1280x720';
% cam.Autofocus = 'on';
%%
% pause(5)
% img = snapshot(cam, 'immediate');
% imshow(img)

%% init ros
rosshutdown()
rosinit('http://192.168.1.10:11311')
robot = Sawyer();
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');


%% joint reading
% tmp = load("calibrate/T.mat");
% Toffset = tmp.T;
% 
% robot.bodyNames{realSenseInd}
gripperBaseInd = 18;
while 1
    msg = joint_sub.LatestMessage;
    if isempty(msg)
        continue
    end
    robot.setJointsMsg(msg);
    hold off
    robot.plotObject
    hold on
% 
    T = robot.getBodyTransform(gripperBaseInd);
    %     T = T*Tc1_c2;
%     T = T*Toffset;
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