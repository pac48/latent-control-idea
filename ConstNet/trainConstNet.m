function [Tall, constNet] = trainConstNet(constNet, imgs, objects, thresh)
setDetectObjects(constNet, objects)
for object = objects
    findInitMatch(constNet, imgs, object{1})
end


numImages = size(imgs, 4);

initialLearnRate = 9e-1;
decay = 0.0005;
momentum = 0.9;
velocity = [];

iteration = 0;
index = 0;

loss = 1;

while loss > thresh && iteration < 1000
    tic
    if mod(iteration, 50) == 0
        setSkipNerf(constNet, false);
    else
        setSkipNerf(constNet, true);
    end

    [loss, gradients, state, Tall] = dlfeval(@simpleModelGradients, constNet, imgs, objects);
    loss

    learnRate = initialLearnRate/(1 + decay*iteration);
    [constNet, velocity] = sgdmupdate(constNet, gradients, velocity, learnRate, momentum);

    if mod(iteration, 300) == 0
        iteration
        index = mod(index + 1, numImages);
        subplot(1,1,1)
        plotAllCorrespondence(constNet, index+1)
    end

    iteration = iteration + 1;
    toc

end

end