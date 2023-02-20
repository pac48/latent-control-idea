function [loss,gradients, state] = FullTFModelGradients(fullTFNet, inputs, targets, objects)
% X: real scene image
global curInd

loss = 0;
gradients = [];

for ind = 1:length(inputs)
    curInd = ind;
    input = inputs{ind};
    target = targets{ind};

    %     [map, state] = getNetOutput(fullTFNet, img(:,:,:, ind), Z(:, ind));
    [map, state] = getfullTFNetOutput(fullTFNet, input);

    %     plotStuff = false;
    plotStuff = true;
    loss = loss + getAllignmentLoss(map, objects, plotStuff, ind, length(inputs));

    loss = loss + getPSMLoss(map, target);

end
loss = loss/9;

if loss ==0
    return
end
gradients = dlgradient(loss, fullTFNet.Learnables);
loss = double(loss);

end
