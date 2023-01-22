function [loss,gradients, state] = simpleModelGradients(dlnet, img, objects)
% X: real scene image

loss = 0;
gradients = [];

for ind = 1:size(img,4)
    [map, state] = getNetOutput(dlnet, img(:,:,:, ind), dlarray(ind,'CB'));

%     plotStuff = false;
    plotStuff = true;
    loss = loss + getAllignmentLoss(map, objects, plotStuff, ind, size(img,4));
end
loss = loss/9;

if loss ==0
    return
end
gradients = dlgradient(loss, dlnet.Learnables);
loss = double(loss);

end