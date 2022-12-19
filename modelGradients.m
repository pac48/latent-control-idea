function [loss,gradients,state] = modelGradients(net, X, mkptsReal)
% X: real scene image

[mkptsNerf, state] = forward(net, X);

loss = sum((mkptsReal - mkptsNerf).^2, 'all');
gradients = dlgradient(loss,net.Learnables);
loss = double(loss);

end