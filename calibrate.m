%% network
data = fileDatastore("transform_data/", "ReadFcn", @load);
allData = readall(data);
X = cellfun(@(x) x.dataPoint.estT, allData, 'UniformOutput', false);
X = reshape([X{:}], 4*4,[]);

Y = cellfun(@(x) x.dataPoint.kineT, allData, 'UniformOutput', false);
Y = reshape([Y{:}], 4*4,[]);

Z = cat(1, X, Y)

dlX = dlarray(Z,'CB');

numFeatures = 16*2;

layers = [
    featureInputLayer(numFeatures);
    TFLayer();
    ];

lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);
%% train
initialLearnRate = 1e-4;
decay = 0.0001;
momentum = 0.9;
velocity = [];
iteration = 0;

% Loop over mini-batches.
while 1
    iteration = iteration + 1;

    [loss, gradients, state] = dlfeval(@modelGradients, dlnet,dlX);
    dlnet.State = state;
    loss
    % Determine learning rate for time-based decay learning rate schedule.
    learnRate = initialLearnRate/(1 + decay*iteration);

    % Update the network parameters using the SGDM optimizer.
    [dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);
end

%% post process

T = Tc1_c2;
% T(1:3,1:3) = eul2rotm([pi/2 -pi/2 0])*Tc1_c2(1:3, 1:3);
T(1:3,1:3) = eul2rotm([pi/2 pi/2 0])*Tc1_c2(1:3, 1:3);

% T = eye(4);
% T(2,end) = 0.015;

save('T', 'T')


%%


function showImages(subRealSense, subWristCam, joint_sub, robot)
imgRealSense = rosReadImage(subRealSense.LatestMessage);
imgWristCam = rosReadImage(subWristCam.LatestMessage);
subplot(1,3,1)
imshow(imgRealSense)
subplot(1,3,2)
imshow(imgWristCam)
subplot(1,3,3)
robot.setJointsMsg(joint_sub.LatestMessage);
robot.plotObject
end