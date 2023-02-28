function msgs = convertMatrixToMsgs(robot, bodyNames, init_msg, t, X, Xd)
% robot robot
% robot.setJointsMsg(joint_sub.receive());
% robot.setJointsMsg(joint_sub.receive());
% robot.setJointsMsg(joint_sub.receive());

robot.setJointsMsg(init_msg)

qCur = robot.getJoints();
dt = (t(end) - t(1))/length(t);

msg = rosmessage('intera_core_msgs/JointCommand', 'DataFormat', 'struct');
msg.Names = robot.jointNames;
msg.Mode = msg.VELOCITYMODE;

msgs = repmat(msg, 1, length(t));
% for i = 1:length(t)
ind = 1;
i = 0;
ti = 0;
while ind <= length(t)
    i = i+1;
    x = X(:, ind);
    xd = Xd(:, ind);
    %     x(4:6) = x(4:6)+x(1:3);
    %     x(7:9) = x(7:9)+x(1:3);

    ti = ti +dt; %t(ind);
    xCur = getXCur(robot, bodyNames);

    [JWrist, JFingerL, JFingerR] = robot.getControlPointsJacobians(bodyNames);
    %     J = cat(1, JWrist(1:3, :), JFingerL(1:3, :), JFingerR(1:3, :));
    weight = 0.1;
    J = cat(1, JWrist(1:3, :), weight*JFingerL(1:3, :), weight*JFingerR(1:3, :));

    error = x - xCur;
    value = xd + 20*error;

    A = J;
    value = reshape(value, 3, []);
    value = value.*[1 weight weight];
    value = reshape(value, [], 1);

    b = double(value);
    reg = .001*eye(size(J,2));
    A = cat(1,A, reg);
    b = cat(1, b, zeros(size(reg,1),1));
    qd = lsqlin(A, b);

    error = x - xCur;
    %     value = xd + 5*error;
    %     qd = pinv(J)*value;
    totalError = norm(error)



    if max(abs(qd)) > .8
        qd = .8*qd./max(abs(qd));
    end

    msgs(i) = msg;
    msgs(i).Velocity = qd;
    msgs(i).Position = qCur;

    robot.setJointsMsg(msgs(i));
    qCur = robot.getJoints() + qd*dt;

    Sec = floor(ti);
    msgs(i).Header.Stamp.Sec = uint32(Sec);
    msgs(i).Header.Stamp.Nsec = uint32((ti - Sec)*1E9);

    if totalError < .05 || norm(qd) < .2
        ind = ind+1;
    end

end

return

% dt = mean(diff(t));
error = 1;
% while norm(error) > .05

% x(4:6) = x(4:6)+x(1:3);
% x(7:9) = x(7:9)+x(1:3);

for it = 1:50
    ti = ti + dt;
    i = i + 1;

    msgs(i) = msg;

    xCur = getXCur(robot, bodyNames);

    [JWrist, JFingerL, JFingerR] = robot.getControlPointsJacobians(bodyNames);
    J = cat(1, JWrist(1:3, :), JFingerL(1:3, :), JFingerR(1:3, :));

    %     JWrist = robot.getControlPointsJacobians(bodyNames(1));
    %     J = cat(1, JWrist(1:3, :));

    error = x - xCur;

    %     xd(1:3) = .5*xd;
    value = 0*xd + 1*error;
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