function [loss, gradients, state] = graspTFModelGradients(graspTFNet, Tbackground_cam, Tcam_obj, Trobot_grasp, Tgrasp_robot, pointsAxis)
% X: real scene image


% [T1, T2, ~, ~, state] = graspTFNet.predict(Tobj_cam, Trobot_grasp);
[Trobot_graspPred, Trobot_cam, Tcam_grasp, Tgrasp_robotPred, state] = graspTFNet.predict(Tbackground_cam, Tcam_obj);


% loss = sum((T1(1:3, 1:3,:)-T2(1:3, 1:3,:)).^2, 'all');
% loss = loss + 10*sum((T1(1:3, end,:)-T2(1:3, end,:)).^2, 'all');

% loss = sum((T1(1:3, 1:3,:)-T2(1:3, 1:3,:)).^2, 'all');
% loss = 10*sum((Trobot_graspPred(1:3,1:4,:) - Trobot_grasp(1:3,1:4,:)).^2, 'all');

% pointsAxis = rand(3, 100);
% % pointsAxis = eye(3);
% pointsAxis(end+1, :) = 1;
% pointsAxis = repmat(pointsAxis, 1,1,size(Trobot_grasp,3));

loss = getTFLoss(Trobot_graspPred, pointsAxis, Trobot_grasp);
loss = loss + getTFLoss(Tgrasp_robotPred, pointsAxis, Tgrasp_robot);

gradients = dlgradient(loss, graspTFNet.Learnables);
loss = double(loss);

end

function loss = getTFLoss(Trobot_graspPred, pointsAxis, Trobot_grasp)
    Trobot_graspPredNoDim = stripdims(Trobot_graspPred);
    pointsPred = pagemtimes(Trobot_graspPredNoDim, pointsAxis);
    
    Trobot_graspNoDim = stripdims(Trobot_grasp);
    points = pagemtimes(Trobot_graspNoDim, pointsAxis);
    
    loss = 10*sum((pointsPred - points).^2, 'all');
    loss = loss./size(Trobot_grasp, 3);
end