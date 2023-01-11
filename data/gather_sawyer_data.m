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

%%
while 1
        msg = subImg.LatestMessage
        if isempty(msg)
            continue
        end
        img = rosReadImage(msg);
%     img = cam.snapshot();
    imshow(img)
    drawnow
end

%% velocity
load('allMsg.mat')

pub = rospublisher('/robot/limb/right/joint_command')
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');
% subImg = rossubscriber('/camera/color/image_raw','sensor_msgs/Image', 'DataFormat','struct')


msg = rosmessage('intera_core_msgs/JointCommand')
msg.Names = robot.jointNames
robot.bodyNames{realSenseInd}
count = 0;
tic
delete(fullfile('data', '*'))

allMsgNew = [];

for i = 1:length(allMsg)
    jointMsgGoal = allMsg(i);
    robot.setJointsMsg(jointMsgGoal);
    qGoal = robot.getJoints();

    q = qGoal*0;
    while 1
        if all(qGoal==q)
            why
        end
        if norm(qGoal-q) < .2
            break
        end

        jointMsg = joint_sub.LatestMessage;
        if isempty(jointMsg) || length(jointMsg.Name) <8
            continue
        end
        robot.setJointsMsg(jointMsg);
        q = robot.getJoints();

        msg.Mode = msg.VELOCITYMODE;
        msg.Velocity = 4*(qGoal - q);

        maxSpeed = .2;
        if any(abs(msg.Velocity)>maxSpeed)
            msg.Velocity = msg.Velocity./max(abs(msg.Velocity));
            msg.Velocity = maxSpeed*msg.Velocity;
        end
        if toc > .2
            %             pause(1)
            'take pic'
                        img = rosReadImage(subImg.receive);
% img = cam.snapshot();
            jointMsg = joint_sub.receive;
            allMsgNew = cat(1, allMsgNew, jointMsg);
            robot.setJointsMsg(jointMsg);
                        imshow(img)
            T = robot.getBodyTransform(realSenseInd);
            dataPoint = struct('img', img, 'T', T);
            save(['data/dataPoint_' num2str(count)],'dataPoint')
            count = count + 1;
            tic
            drawnow
            length(allMsgNew)

        end

        pub.send(msg)

    end
    i

end

% save('allMsgNew.mat', 'allMsgNew')

%% position
load('allMsgNew.mat')

pub = rospublisher('/robot/limb/right/joint_command')
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');
% subImg = rossubscriber('/camera/color/image_raw','sensor_msgs/Image', 'DataFormat','struct')


msg = rosmessage('intera_core_msgs/JointCommand')
msg.Names = robot.jointNames
robot.bodyNames{realSenseInd}
count = 0;
tic
delete(fullfile('data', '*'))

compareInds = cellfun(@(x) ~strcmp(x, 'head_pan'), robot.jointNames) & ...
    cellfun(@(x) ~strcmp(x, 'right_gripper_l_finger_joint'), robot.jointNames) & ...
cellfun(@(x) ~strcmp(x, 'right_gripper_r_finger_joint'), robot.jointNames);

% allMsgNew = [];

for i = 1:length(allMsgNew)
    jointMsgGoal = allMsgNew(i);
    robot.setJointsMsg(jointMsgGoal);
    qGoal = robot.getJoints();

    q = qGoal*0;
    while 1
        if all(qGoal==q)
            why
        end
        if norm(qGoal(compareInds)-q(compareInds)) < .01
%             pause(1)
            tic
            while toc < 1
                msg.Mode = msg.POSITIONMODE;
                msg.Velocity = qGoal;
                pub.send(msg)
            end
            'take pic'
                        img = rosReadImage(subImg.receive);
%             img = cam.snapshot();

            jointMsg = joint_sub.receive;
            %             allMsgNew = cat(1, allMsgNew, jointMsg);
            robot.setJointsMsg(jointMsg);
            imshow(img)
            T = robot.getBodyTransform(realSenseInd);
            dataPoint = struct('img', img, 'T', T)
            save(['data/dataPoint_' num2str(count)],'dataPoint')
            count = count + 1;
            tic
            drawnow
            break
        end

        jointMsg = joint_sub.receive();
        if isempty(jointMsg) || length(jointMsg.Name) < 8
            continue
        end
        robot.setJointsMsg(jointMsg);
        q = robot.getJoints();

        msg.Mode = msg.POSITIONMODE;
        msg.Position = qGoal;
        pub.send(msg)

    end

    i

end

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