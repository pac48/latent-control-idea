function [loss,gradients, state] = simpleModelGradients(dlnet, img, objects)
% X: real scene image

loss = 0;

for ind = 1:size(img,4)
    [map, state] = getNetOutput(dlnet, img(:,:,:, ind), dlarray(ind,'CB'));

%     plotStuff = false;
    plotStuff = true;
    loss = loss + getAllignmentLoss(map, objects, plotStuff, ind, size(img,4));
end
loss = loss/9;

% loss = loss./(size(img,4)*length(objects));
% if loss > 5
%     loss = loss/10;
% end
% loss = 3*tanh(loss/100);

gradients = dlgradient(loss, dlnet.Learnables);
loss = double(loss);

end