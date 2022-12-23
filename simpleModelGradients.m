function [loss,gradients, state] = simpleModelGradients(dlnet, img)
% X: real scene image

[realPoints2D, nerfPoints2D, state] = dlnet.predict(img);
% inds = extractdata(isfinite(nerfPoints2D) & isfinite(realPoints2D));
realPoints2D = dlarray(gather(extractdata(realPoints2D)), 'SSB');

% inds = (1:5) + randi(size(nerfPoints2D,2)-5);
% nerfPoints2D = nerfPoints2D(:,inds,:);
% realPoints2D = realPoints2D(:,inds,:);

loss = 1000*sum( (nerfPoints2D - realPoints2D).^2, 'all');
% loss = 1000*max( (nerfPoints2D - realPoints2D).^2,[],"all");

loss = loss./size(realPoints2D,2);
if numel(realPoints2D) <= 2
    loss = 0*loss;
end


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