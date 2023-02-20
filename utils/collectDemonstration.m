function allMsg = collectDemonstration(robot, joint_sub, timeout)
allMsg = [];

robot.setJointsMsg(receive(joint_sub))
q = robot.getJoints();

count = 0;
tic
while toc < timeout
    jointMsg = joint_sub.LatestMessage;
    robot.setJointsMsg(jointMsg);
    if (norm(q-robot.getJoints()) < .01 )
        continue
    end
    q = robot.getJoints();
    allMsg = cat(1, allMsg, joint_sub.LatestMessage);
    count = count + 1;
end

deleteInds = [];
for i = 1:length(allMsg)
    allMsg(i).Position
    if length(allMsg(i).Position) < 7    
        deleteInds = cat(1, deleteInds, i);
    end
end

allMsg(deleteInds) = [];

end