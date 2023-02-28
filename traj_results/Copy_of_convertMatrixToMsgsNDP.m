function msgs = convertMatrixToMsgsNDP(robot, bodyNames, init_msg, t, X, Xd)
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
for i = 1:length(t)
    x = X(:, i);
    xd = Xd(:, i);
%     x(4:6) = x(4:6)+x(1:3);
%     x(7:9) = x(7:9)+x(1:3);

    ti = t(i);
    xCur = getXCur(robot, bodyNames);

    [JWrist, JFingerL, JFingerR] = robot.getControlPointsJacobians(bodyNames);
    weight = 0.05;
    J = cat(1, JWrist(1:3, :), weight*JFingerL(1:3, :), weight*JFingerR(1:3, :));

    error = x - xCur;
    value = xd + 6*error;

     A = J;
     value = reshape(value, 3, []);
     value = value.*[1 weight weight];
     value = reshape(value, [], 1);
            
     b = double(value);
     reg = .01*eye(size(J,2));
     A = cat(1,A, reg);
     b = cat(1, b, zeros(size(reg,1),1));
     qd = lsqlin(A, b);
%     qd = pinv(J)*value;
    norm(error)


% %     posRef = x(1:3);
% %     dir1Ref = x(4:6);%(x(4:6) - posRef);
% %     dir2Ref = x(7:9);%(x(7:9) - posRef);
% %     dir1Ref = dir1Ref./norm(dir1Ref);
% %     dir2Ref = dir2Ref./norm(dir2Ref);
% % 
% %     posCur = xCur(1:3); 
% %     dir1Cur = (xCur(4:6) - posCur);
% %     dir2Cur = (xCur(7:9) - posCur);
% %     dir1Cur = dir1Cur./norm(dir1Cur);
% %     dir2Cur = dir2Cur./norm(dir2Cur);
% % 
% %     errorPos = xd(1:3) + 4*(posRef - posCur);
% %     
% %     errorAxis1 =  cross(dir1Ref, dir1Cur);
% %     errorAxis1 = errorAxis1./norm(errorAxis1);
% %     errorDir1 = acos(dot(dir1Ref,dir1Cur)); % angle rad
% %     errorAxis1 =  -20*errorDir1*errorAxis1;
% % 
% % 
% %     errorAxis2 =  cross(dir2Ref, dir2Cur);
% %     errorAxis2 = errorAxis2./norm(errorAxis2);
% %     errorDir2 = acos(dot(dir2Ref,dir2Cur)); % angle rad
% %     errorAxis2 =  -20*errorDir2*errorAxis2;
% % 
% %      A = cat(1, JWrist(1:3, :), JWrist(4:6, :), JWrist(4:6, :));
% %      b = cat(1, errorPos, errorAxis1, errorAxis2);
% % 
% % %      A = cat(1, JWrist(1:3, :), JWrist(4:6, :));
% % %      b = cat(1, errorPos, zeros(3,1));
% % 
% %      reg = .01*eye(size(JWrist,2));
% %      A = cat(1,A,reg);
% %      b = cat(1, b, zeros(size(reg,1),1));
% % 
% %      b = double(b);
% %      qd = lsqlin(A, b);

% %     norm((posRef - posCur))


    if max(qd) > 10
        qd = 10*qd./max(qd);
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

% x(4:6) = x(4:6)+x(1:3);
% x(7:9) = x(7:9)+x(1:3);

% % for it = 1:50
% %     ti = ti + dt;
% %     i = i + 1;
% %     
% %     msgs(i) = msg;
% % 
% %     xCur = getXCur(robot, bodyNames);
% % 
% %     [JWrist, JFingerL, JFingerR] = robot.getControlPointsJacobians(bodyNames);
% %     J = cat(1, JWrist(1:3, :), JFingerL(1:3, :), JFingerR(1:3, :));
% %   
% % %     JWrist = robot.getControlPointsJacobians(bodyNames(1));
% % %     J = cat(1, JWrist(1:3, :));
% % 
% %     error = x - xCur;
% % 
% % %     xd(1:3) = .5*xd;
% %     value = 0*xd + 1*error;
% %     qd = pinv(J)*value;
% % 
% %     if max(qd) > .2
% %         qd = .2*qd./max(qd);
% %     end
% % 
% %     msgs(i).Velocity = qd;
% %     msgs(i).Position = qCur;
% %  
% %     robot.setJointsMsg(msgs(i));
% %     qCur = robot.getJoints() + qd*dt;
% % 
% %     Sec = floor(ti);
% %     msgs(i).Header.Stamp.Sec = uint32(Sec);
% %     msgs(i).Header.Stamp.Nsec = uint32((ti - Sec)*1E9);
% % end

end

function Xcur = getXCur(robot, bodyNames)
Xcur = zeros(3*length(bodyNames), 1);
for b = 1:length(bodyNames)
    bodyName = bodyNames{b};
    T = robot.getBodyTransform(bodyName);
    Xcur((1:3) +(b-1)*3) = T(1:3, end);
end
end