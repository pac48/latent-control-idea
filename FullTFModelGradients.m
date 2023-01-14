function [loss,gradients, state] = FullTFModelGradients(FullTFNet, img, Z, objects)
% X: real scene image
global curInd
loss = 0;

for ind = 1:size(img,4)
    curInd = ind;
    %     [map, state] = getNetOutput(dlnet, img(:,:,:, ind), dlarray(ind,'CB'));
    [map, state] = getNetOutput(FullTFNet, img(:,:,:, ind), Z(:, ind));

    for i = 1:length(objects)
        object = objects{i};

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

gradients = dlgradient(loss, FullTFNet.Learnables);
loss = double(loss);

% hold off
% x = cat(1, nerfPoints2D(1,:), realPoints2D(1,:));
% y = cat(1, nerfPoints2D(2,:), realPoints2D(2,:));
% plot(x,y)
% hold on
% plot(realPoints2D(1,:), realPoints2D(2,:),'LineStyle','none', 'Marker','.', 'MarkerSize',8)
% drawnow
end
