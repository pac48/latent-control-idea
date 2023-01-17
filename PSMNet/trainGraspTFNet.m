function [graspTFNet] = trainGraspTFNet(graspTFNet, Tbackground_cam, Tcam_obj, Trobot_grasp, Tgrasp_robot, thresh)

initialLearnRate = 1e-5;
decay = 0.00005;
momentum = 0.9999;
velocity = [];

iteration = 0;
loss = 1;

pointsAxis = rand(3, 1000);
% pointsAxis = eye(3);
pointsAxis(end+1, :) = 1;
pointsAxis = repmat(pointsAxis, 1,1,size(Trobot_grasp,3));


while loss > thresh && iteration < 4000
    tic

    [loss, gradients, state] = dlfeval(@graspTFModelGradients, graspTFNet, Tbackground_cam, Tcam_obj, Trobot_grasp,Tgrasp_robot, pointsAxis);
    loss

    learnRate = initialLearnRate/(1 + decay*iteration);
    [graspTFNet, velocity] = sgdmupdate(graspTFNet, gradients, velocity, learnRate, momentum);

    iteration = iteration + 1;
    toc

end

end