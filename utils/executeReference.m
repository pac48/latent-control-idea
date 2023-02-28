function executeReference(gain, allMsg, joint_sub, robot, pubJointCmd,pubGripper, time_scale, th, is_reverse)
% Take a JointState trajectory and execute it using velocity command.
% allMsg: trajectory
% joint_sub:
% robot: object
% pubJointCmd: publisher
% pubGripper: publisher
% time_scale: scale
% th: velocity threshold so that robot doesn't make sudden move.



msgJointCmd = rosmessage('intera_core_msgs/JointCommand', 'DataFormat', 'struct');

msgJointCmd.Mode = msgJointCmd.VELOCITYMODE;

inds = [1 3 4 5 6 7 8];
msgJointCmd.Names = robot.jointNames(inds);

% msgGripper =  rosmessage(pubGripper);  %get pubGripper first

time = arrayfun(@(x) double(x.Header.Stamp.Sec) + double(x.Header.Stamp.Nsec)*1E-9, allMsg);
time = time - time(1);
refTraj = allMsg;

if is_reverse
    refTraj=allMsg(end:-1:1);
end

% arrayfun(@(x) x.Position, allMsg, 'UniformOutput', false)
for i = 2:length(refTraj) %was 1
    if length(refTraj(i).Position) < 7
        refTraj(i) = refTraj(i-1);
    end
end


% gripper_status=allMsg(1).gripper.Data;

time = time*time_scale;

vel = 1;
tic
while (norm(vel) > .002 && toc < time(end) + 2) || toc< time(end)
    msg = joint_sub.receive();
    if isempty(msg) || length(msg.Position) < 7
        continue
    end


    curTime =  toc;
    ind = find(curTime < time, 1);
    if isempty(ind)
        ind = length(time);
    end

    robot.setJointsMsg(msg);
    curJoint = robot.getJoints();
    curJoint = curJoint(inds);

    refmsg = refTraj(ind);

    %     grip=refmsg.gripper.Data;
    %     if grip~=gripper_status   %gripper status changed
    %            msgGripper.Data = grip;
    %            send(pubGripper, msgGripper)
    %     end
    %     gripper_status=grip; %set prev
    %

    robot.setJointsMsg(refmsg);
    refJoint = robot.getJoints();
    refJoint = refJoint(inds);

    refVel = robot.getJointsVelfromMsg(refmsg);
    refVel = refVel(inds)./time_scale;

    refVel = refVel*(ind<length(time));
    vel = refVel + gain*(refJoint - curJoint);

    %ensure vel isn't too high. parameter to play=th
    if max(abs(vel) > th)
        vel = th*vel./max(abs(vel));
    end

    msgJointCmd.Velocity = vel;
    send(pubJointCmd, msgJointCmd)

end

end