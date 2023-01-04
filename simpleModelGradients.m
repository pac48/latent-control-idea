function [loss,gradients, state] = simpleModelGradients(dlnet, img)
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

[backgroundTF, backgroundTFConst, ...
    backgroundRealPoints2D, backgroundNerfPoints2D, ...
    cupTF, cupTFConst, ...
    cupRealPoints2D, cupNerfPoints2D, ...
    boxTF, boxTFConst, ...
    boxRealPoints2D, boxNerfPoints2D, ...
    state] = dlnet.predict(img);

loss = 0;

if numel(backgroundRealPoints2D) > 2
    backgroundRealPoints2D = dlarray(gather(extractdata(backgroundRealPoints2D)), 'SSB');
    backgroundLoss = 1000*sum( (backgroundNerfPoints2D - backgroundRealPoints2D).^2, 'all');
    backgroundLoss = backgroundLoss./size(backgroundRealPoints2D,2);
    loss = loss + backgroundLoss;
    %     backgroundLoss = 0*backgroundLoss;
end

if numel(cupRealPoints2D) > 2
    cupRealPoints2D = dlarray(gather(extractdata(cupRealPoints2D)), 'SSB');
    cupLoss = 100*sum( (cupNerfPoints2D - cupRealPoints2D).^2, 'all');
    cupLoss = cupLoss./size(cupRealPoints2D,2);
    loss = loss + cupLoss;
    %     cupLoss = 0*cupLoss;
end

if numel(boxRealPoints2D) > 2
    boxRealPoints2D = dlarray(gather(extractdata(boxRealPoints2D)), 'SSB');
    boxLoss = 100*sum( (boxNerfPoints2D - boxRealPoints2D).^2, 'all');
    boxLoss = boxLoss./size(boxRealPoints2D,2);
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