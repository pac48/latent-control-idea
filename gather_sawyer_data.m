
robot = Sawyer();


%% joint reading
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');
robot.bodyNames{23}
while 1
    msg = joint_sub.LatestMessage;
    if isempty(msg)
        continue
    end
    robot.setJointsMsg(msg);
    hold off
    robot.plotObject
    hold on

    T = robot.getBodyTransform(23);
    point = T(1:3,end);

    line =  [point point+T(1:3,1)*.2];

    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.')

    drawnow

end
%%
allMsg = [];
allMsg = cat(1, allMsg, joint_sub.LatestMessage)
%%
load('allMsg.mat')

for i = 1:length(allMsg)
    robot.setJointsMsg(allMsg(i));
    hold off
    robot.plotObject
    hold on

    T = robot.getBodyTransform(23);
    point = T(1:3,end);

    line =  [point point+T(1:3,1)*.2];

    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.')

    drawnow
    pause(1)
end


%%
subImg = rossubscriber('/camera/color/image_raw','sensor_msgs/Image', 'DataFormat','struct')
while 1
    msg = subImg.LatestMessage
    if isempty(msg)
        continue
    end
    img = rosReadImage(msg);
    imshow(img)
    drawnow
end

%%
load('allMsg.mat')

pub = rospublisher('/robot/limb/right/joint_command')
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');

msg = rosmessage('intera_core_msgs/JointCommand')
msg.Names = robot.jointNames
robot.bodyNames{23}
count = 0;
tic

for i = 1:length(allMsg)
    jointMsgGoal = allMsg(i);
    robot.setJointsMsg(jointMsgGoal);
    qGoal = robot.getJoints();

    q = qGoal*0;
    while 1
        if all(qGoal==q)
            why
        end
        if norm(qGoal-q) < .01
            break
        end

        jointMsg = joint_sub.LatestMessage;
        if isempty(jointMsg) || length(jointMsg.Name) <8
            continue
        end
        robot.setJointsMsg(jointMsg);
        q = robot.getJoints();


        msg.Mode = msg.VELOCITYMODE;
        %         msg.Velocity = .6*(qGoal - q) + .1*(qGoal - q)./(norm((qGoal - q))+0.001);
        msg.Velocity = 2*(qGoal - q);
        maxSpeed = .25;
        if any(abs(msg.Velocity)>maxSpeed)
            msg.Velocity = msg.Velocity./max(abs(msg.Velocity));
            msg.Velocity = maxSpeed*msg.Velocity;
        end
        if toc > 1.25
            pause(1)
            'take pic'
            img = rosReadImage(subImg.LatestMessage);
            T = robot.getBodyTransform(23);
            dataPoint = struct('img', img, 'T', T)
            save(['data/dataPoint_' num2str(count)],'dataPoint')
            count = count + 1;
            tic
        end

        pub.send(msg)

    end

    i

end



