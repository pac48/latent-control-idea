function [loss,gradients, state] = FullTFModelGradients(fullTFNet, img, Z)
% X: real scene image
global curInd

loss = 0;

objects = getObjects(fullTFNet);

for ind = 1:size(img,4)
    curInd = ind;
    %     [map, state] = getNetOutput(dlnet, img(:,:,:, ind), dlarray(ind,'CB'));
    [map, state] = getNetOutput(fullTFNet, img(:,:,:, ind), Z(:, ind));

    for i = 1:length(objects)
        object = objects{i};

        T = getObjectTransforms(fullTFNet, map, object);
        if isempty(T)
            continue
        end

        [mkptsNerf, mkptsReal] = getObjectPoints(map, object);

        if numel(mkptsReal) > 2 && strcmp(object, 'background')
            loss = loss + getObjectLoss(mkptsReal, mkptsNerf, 10, 1000);
            %         loss = loss + backgroundLoss;
        elseif numel(mkptsReal) > 2
            loss = loss + getObjectLoss(mkptsReal, mkptsNerf, 10, 1000);
            %         loss = loss + cupLoss;
        end

    end
end

loss = loss./(size(img,4)*length(objects));

gradients = dlgradient(loss, fullTFNet.Learnables);
loss = double(loss);

end
