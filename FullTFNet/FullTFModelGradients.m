function [loss,gradients, state] = FullTFModelGradients(fullTFNet, img, Z, objects)
% X: real scene image
global curInd

loss = 0;
gradients = [];

for ind = 1:size(img,4)
    curInd = ind;

%     [map, state] = getNetOutput(fullTFNet, img(:,:,:, ind), Z(:, ind));
     [map, state] = getfullTFNetOutput(fullTFNet, img(:,:,:, ind), Z(:, ind));

%     plotStuff = false;
    plotStuff = true;
    loss = loss + getAllignmentLoss(map, objects, plotStuff, ind, size(img,4));
end
loss = loss/9;

if loss ==0
    return
end
gradients = dlgradient(loss, fullTFNet.Learnables);
loss = double(loss);

end
