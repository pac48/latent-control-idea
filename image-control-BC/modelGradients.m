function [loss,gradients,state] = modelGradients(net, X, Y)

[Y_pred,state] = forward(net, X);

loss = sum(( Y_pred - Y(1:2,:,:)).^2 ,'all');
loss = loss./size(Y,2);
% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);


loss = double(loss);

end