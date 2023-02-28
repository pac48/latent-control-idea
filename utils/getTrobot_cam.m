function Trobot_cam2 = getTrobot_cam(robot, msg)
camLinkInd = 24; % cam link
robot.setJointsMsg(msg);

Trobot_cam2 = robot.getBodyTransform(camLinkInd);
Trobot_cam2(1:3, end) = Trobot_cam2(1:3, end) + .02*Trobot_cam2(1:3, 2);


% rotX = -90*pi/180;
% Trobot_cam2(1:3, 1:3) = axang2rotm([1 0 0 rotX]);

% return
% rotZ = 0*pi/180;
% Trobot_cam2(1:3, 1:3) = Trobot_cam2(1:3, 1:3)*axang2rotm([Trobot_cam2(1:3, 3)' rotZ])
% rotY = -0*pi/180;
% Trobot_cam2(1:3, 1:3) = Trobot_cam2(1:3, 1:3)*axang2rotm([Trobot_cam2(1:3, 2)' rotY])
rotX = 5*pi/180;
Trobot_cam2(1:3, 1:3) = Trobot_cam2(1:3, 1:3)*axang2rotm([1 0 0 rotX]);
end