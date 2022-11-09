function [loss,gradients,state] = modelGradients(net, X, Y)

% Forward data through network.
[Y_pred,state] = forward(net, X);

% Calculate cross-entropy loss.
loss = sum(abs(Y_pred-Y),'all');

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);


loss = double(loss);

end