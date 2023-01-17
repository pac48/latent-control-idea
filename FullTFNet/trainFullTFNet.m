function trainFullTFNet(fullTFNet, imgs, objects, thresh)
numImages = size(imgs, 4);
setDetectObjects(fullTFNet, objects)

initialLearnRate = 2e-4;
decay = 0.005;
momentum = 0.95;
velocity = [];

iteration = 0;
index = 0;

loss = 1;

while loss > thresh
    tic
    if mod(iteration, 100) == 0
        setSkipNerf(fullTFNet, false);
    else
        setSkipNerf(fullTFNet, true);
    end

    [loss, gradients, state] = dlfeval(@FullTFModelGradients, fullTFNet, imgs, Z); % need to change  simpleModelGradients
    loss
    %     dlnet.State = state;
    learnRate = initialLearnRate/(1 + decay*iteration);
    [fullTFNet,velocity] = sgdmupdate(fullTFNet, gradients, velocity, learnRate, momentum);

    if mod(iteration, 100) == 0
        iteration
        index = mod(index + 1, numImages);
        subplot(1,1,1);
        plotAllCorrespondence(fullTFNet, index+1)
    end

    iteration = iteration + 1;
    toc
end

end