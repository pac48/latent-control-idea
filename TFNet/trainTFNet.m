function TFNet = trainTFNet(TFNet, allZ, Tall, thresh)
initialLearnRateTF = 9e-3;
decayTF = 0.000005;
momentumTF = 0.99;
velocityTF = [];
iteration = 0;

noiseLevel = 0.00;
loss =  1;

while loss > thresh
    tic
    [loss, gradients, ~] = dlfeval(@simpleTFModelGradients, TFNet, allZ + noiseLevel*(rand(size(allZ))-.5), Tall);
    loss
    learnRateTF = initialLearnRateTF/(1 + decayTF*iteration);
    [TFNet, velocityTF] = sgdmupdate(TFNet, gradients, velocityTF, learnRateTF, momentumTF);

    iteration = iteration + 1;
    toc

end
[fileNames, file_path] = TFGetDataPath();
file_path = file_path{1};
ind = find(file_path == '/', 9999);
ind  = ind(end);
file_path = file_path(1:ind-1); 

% save(fullfile(file_path, 'TFNetPretrained'), 'TFNet')

end