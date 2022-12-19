function [loss,gradients, state] = simpleModelGradients(dlnet, feature, img)
% X: real scene image

[realPoints2D, nerfPoints2D, state] = dlnet.forward(feature, img);
% inds = extractdata(isfinite(nerfPoints2D) & isfinite(realPoints2D));
realPoints2D = dlarray(gather(extractdata(realPoints2D)), 'SSB');

loss = sum( (nerfPoints2D - realPoints2D).^2, 'all');
loss = 1000*loss./numel(realPoints2D);
if numel(realPoints2D) < 20
    loss = 0*loss;
end


gradients = dlgradient(loss, dlnet.Learnables);
loss = double(loss);

hold off
x = cat(1, nerfPoints2D(1,:), realPoints2D(1,:));
y = cat(1, nerfPoints2D(2,:), realPoints2D(2,:));
plot(x,y)
drawnow
end