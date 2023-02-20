function msgs = convertMatrixToMsgs(robot, bodyNames, joint_sub, t, X, Xd)
% robot robot
robot.setJointsMsg(joint_sub.receive());
robot.setJointsMsg(joint_sub.receive());
robot.setJointsMsg(joint_sub.receive());

qCur = robot.getJoints();
dt = (t(end) - t(1))/length(t);

msg = rosmessage('intera_core_msgs/JointCommand', 'DataFormat', 'struct');
msg.Names = robot.jointNames;
msg.Mode = msg.VELOCITYMODE;

msgs = repmat(msg, 1, length(t));
for i = 1:length(t)
    x = X(:, i);
    xd = Xd(:, i);
    ti = t(i);
    xCur = getXCur(robot, bodyNames);

    [JWrist, JFingerL, JFingerR] = robot.getControlPointsJacobians(bodyNames);
    J = cat(1, JWrist(1:3, :), JFingerL(1:3, :), JFingerR(1:3, :));

    pos = x(1:3);
    dir1 = (x(4:6) - pos);
    dir2 = (x(7:9) - pos);
    dir1 = dir1./norm(dir1);
    dir2 = dir2./norm(dir2);

    posCur = xCur(1:3); 
    dir1Cur = (xCur(4:6) - posCur);
    dir2Cur = (xCur(7:9) - posCur);
    dir1Cur = dir1Cur./norm(dir1Cur);
    dir2Cur = dir2Cur./norm(dir2Cur);


    errorPos = xd(1:3) + 1*(pos - posCur);
    
    errorDir1 = acos(dot(dir1,dir1Cur)); % angle rad
    errorAxis1 =  -.4*errorDir1*cross(dir1, dir1Cur);
    
    errorDir2 = acos(dot(dir1,dir1Cur)); % angle rad
    errorAxis2 =  -.4*errorDir2*cross(dir2, dir2Cur);


     A = cat(1, JWrist(1:3, :), JWrist(4:6, :), JWrist(4:6, :));
     b = cat(1, errorPos, errorAxis1, errorAxis2);

%      A = cat(1, JWrist(1:3, :), JWrist(4:6, :));
%      b = cat(1, errorPos, errorAxis1);


     b = double(b);
     qd = lsqlin(A, b);

    norm(errorPos)

%     value = xd.*[1;1;1;0;0;0;0;0;0] + 1*error.*[1;1;1;  0;0;0;  0;0;0];
%     qd = pinv(J)*value;

%     inv(J'*J + .01*eye(size(J,2) ))*J'*value;
%      weight = [100;100;100; .1;.1;.1; .1;.1;.1];
%      weight = [1;1;1; 1;1;1; 1;1;1];
%     weight = [1;1;1; 0.01;0.01;0.01;  0.00001;0.00001;0.00001];

%      A = double(cat(1, weight.*J, .00001*eye(size(J,2) )) );
%      b = double(cat(1, weight.*value, zeros(size(J,2),1)));



    if max(qd) > .2
        qd = .2*qd./max(qd);
    end

    msgs(i).Velocity = qd;
    msgs(i).Position = qCur;
 
    robot.setJointsMsg(msgs(i));
    qCur = robot.getJoints() + qd*dt;

    Sec = floor(ti);
    msgs(i).Header.Stamp.Sec = uint32(Sec);
    msgs(i).Header.Stamp.Nsec = uint32((ti - Sec)*1E9);

end

% dt = mean(diff(t));
error = 1;
% while norm(error) > .05
for it = 1:500
    ti = ti + dt;
    i = i + 1;
    
    msgs(i) = msg;

    xCur = getXCur(robot, bodyNames);

    [JWrist, JFingerL, JFingerR] = robot.getControlPointsJacobians(bodyNames);
    J = cat(1, JWrist(1:3, :), JFingerL(1:3, :), JFingerR(1:3, :));

    error = x - xCur;

    xd = .5*xd;
    value = xd + 1*error;
    qd = pinv(J)*value;

    if max(qd) > .2
        qd = .2*qd./max(qd);
    end

    msgs(i).Velocity = qd;
    msgs(i).Position = qCur;
 
    robot.setJointsMsg(msgs(i));
    qCur = robot.getJoints() + qd*dt;

    Sec = floor(ti);
    msgs(i).Header.Stamp.Sec = uint32(Sec);
    msgs(i).Header.Stamp.Nsec = uint32((ti - Sec)*1E9);
end

end

function Xcur = getXCur(robot, bodyNames)
Xcur = zeros(3*length(bodyNames), 1);
for b = 1:length(bodyNames)
    bodyName = bodyNames{b};
    T = robot.getBodyTransform(bodyName);
    Xcur((1:3) +(b-1)*3) = T(1:3, end);
end
end