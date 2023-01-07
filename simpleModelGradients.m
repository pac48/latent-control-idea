function [loss,gradients, state] = simpleModelGradients(dlnet, Z, img)
% X: real scene image

backgroundRealPoints2D = [];
cupRealPoints2D = [];
boxRealPoints2D = [];

% [realPoints2D, nerfPoints2D, state] = dlnet.predict(img);

% [backgroundTF, backgroundTFConst, ...
%     backgroundRealPoints2D, backgroundNerfPoints2D, ...
%     cupTF, cupTFConst, ...
%     cupRealPoints2D, cupNerfPoints2D, ...
%     state] = dlnet.predict(img);

% [backgroundTF, backgroundTFConst, ...
%     backgroundRealPoints2D, backgroundNerfPoints2D, ...
%     cupTF, cupTFConst, ...
%     cupRealPoints2D, cupNerfPoints2D, ...
%     boxTF, boxTFConst, ...
%     boxRealPoints2D, boxNerfPoints2D, ...
%     state] = dlnet.predict(img);

% [backgroundTF, ...
%     backgroundRealPoints2D, backgroundNerfPoints2D, ...
%     cupTF, ...
%     cupRealPoints2D, cupNerfPoints2D, ...
%     boxTF, ...
%     boxRealPoints2D, boxNerfPoints2D, ...
%     state] = dlnet.predict(Z, img);

[backgroundTF, ...
    backgroundRealPoints2D, backgroundNerfPoints2D, ...
    boxTF, ...
    boxRealPoints2D, boxNerfPoints2D, ...
    state] = dlnet.predict(Z, img);

% [backgroundTF, ...
%     backgroundRealPoints2D, backgroundNerfPoints2D, ...
%     state] = dlnet.predict(Z, img);

loss = 0;

if numel(backgroundRealPoints2D) > 2
    backgroundLoss = getObjectLoss(backgroundRealPoints2D, backgroundNerfPoints2D, 1, 10);
    loss = loss + backgroundLoss;

    %     backgroundRealPoints2D = dlarray(gather(extractdata(backgroundRealPoints2D)), 'SSB');
    %     backgroundLoss = 100*sum( (backgroundNerfPoints2D - backgroundRealPoints2D).^2, 'all');
    %     backgroundLoss = backgroundLoss./size(backgroundRealPoints2D,2);
    %     loss = loss + backgroundLoss;
    %     backgroundLoss = 0*backgroundLoss;
end

if numel(cupRealPoints2D) > 2
    cupLoss = getObjectLoss(cupRealPoints2D, cupNerfPoints2D, 10, 1000);
    loss = loss + cupLoss;

    %     cupRealPoints2D = dlarray(gather(extractdata(cupRealPoints2D)), 'SSB');
    %     cupLoss = 100*sum( (cupNerfPoints2D - cupRealPoints2D).^2, 'all');
    %     cupLoss = cupLoss./size(cupRealPoints2D,2);
    %     loss = loss + cupLoss;
    %     cupLoss = 0*cupLoss;
end

if numel(boxRealPoints2D) > 2
    boxLoss = getObjectLoss(boxRealPoints2D, boxNerfPoints2D, 10, 1000);
    %     boxRealPoints2D = dlarray(gather(extractdata(boxRealPoints2D)), 'SSB');
    %     boxLoss = 100*sum( (boxNerfPoints2D - boxRealPoints2D).^2, 'all');
    %
    %     boxNerfPoints2DZero = boxNerfPoints2D- mean(boxNerfPoints2D,2);
    %     boxRealPoints2DZero = boxRealPoints2D- mean(boxRealPoints2D,2);
    %
    %     boxLoss = boxLoss + 5000*sum( (boxNerfPoints2DZero - boxRealPoints2DZero).^2, 'all');
    %
    %     boxLoss = boxLoss./size(boxRealPoints2D,2);
    loss = loss + boxLoss;
    %     cupLoss = 0*cupLoss;
end


% loss = backgroundLoss + cupLoss;

gradients = dlgradient(loss, dlnet.Learnables);
loss = double(loss);

% hold off
% x = cat(1, nerfPoints2D(1,:), realPoints2D(1,:));
% y = cat(1, nerfPoints2D(2,:), realPoints2D(2,:));
% plot(x,y)
% hold on
% plot(realPoints2D(1,:), realPoints2D(2,:),'LineStyle','none', 'Marker','.', 'MarkerSize',8)
% drawnow
end

function loss = getObjectLoss(realPoints2D, nerfPoints2D, meanPenalty, rotPenalty)
realPoints2D = dlarray(gather(extractdata(realPoints2D)), 'SSB');
nerfPoints2DMean = mean(nerfPoints2D,2);
realPoints2DMean = mean(realPoints2D,2);
nerfPoints2DZero = nerfPoints2D - nerfPoints2DMean;
realPoints2DZero = realPoints2D - realPoints2DMean;

tmp = nerfPoints2DMean - realPoints2DMean;
lmin = min([tmp.^2 abs(tmp)], [],2);
loss = meanPenalty*sum(lmin, 'all');

tmp = mean(abs(nerfPoints2DZero - realPoints2DZero),2);
lmin = min([tmp.^2 abs(tmp)], [],2);
loss = loss + rotPenalty*sum(lmin, 'all');

% loss = loss + rotPenalty*sum( (nerfPoints2DZero - realPoints2DZero).^2, 'all');
%     loss = loss./size(realPoints2D,2);
%     loss = loss + loss;
end