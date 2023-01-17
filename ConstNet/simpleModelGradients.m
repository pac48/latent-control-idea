function [loss,gradients, state, Tall] = simpleModelGradients(dlnet, img, objects)
% X: real scene image

loss = 0;

% objects = getObjects(dlnet);

Tall = containers.Map(objects, repmat({{}}, 1, length(objects)));


for ind = 1:size(img,4)
    [map, state] = getNetOutput(dlnet, img(:,:,:, ind), dlarray(ind,'CB'));
    
    for i = 1:length(objects)
        object = objects{i};

        T = getObjectTransforms(dlnet, map, object);
        Tall(object) = cat(2, Tall(object), {T});
        if isempty(T)
            continue
        end

        [mkptsNerf, mkptsReal] = getObjectPoints(map, object);

        if numel(mkptsReal) > 2 && strcmp(object, 'background')
            loss = loss + getObjectLoss(mkptsReal, mkptsNerf, 10, 1000);
            %         loss = loss + backgroundLoss;
        elseif numel(mkptsReal) > 2
            loss = loss + getObjectLoss(mkptsReal, mkptsNerf, 100, 1000);
            %         loss = loss + cupLoss;
        end

    end
end

loss = loss./(size(img,4)*length(objects));

gradients = dlgradient(loss, dlnet.Learnables);
loss = double(loss);

end