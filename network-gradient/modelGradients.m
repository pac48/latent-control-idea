function [loss,gradients,state] = modelGradients(net, X, Y)

% Forward data through network.
[Y_pred,state] = forward(net, X);

Xgrid = -1:.03:1;
Ygrid = -1:.03:1;
Y_pred = reshape(Y_pred, length(Xgrid), length(Ygrid), 1, []);

[XMesh,YMesh] = meshgrid(-1:.03:1,-1:.03:1);


Ynew = reshape(Y,1,1,2,[]);
% loss = 10-sum(Y_pred.*(exp(-150*sum((cat(3, XMesh,YMesh)-Ynew).^2, 3))) ,'all');
loss = sum( (Y_pred - (exp(-10*sum((cat(3, XMesh,YMesh)-Ynew).^2, 3)))).^2 ,'all');


% Calculate cross-entropy loss.
% loss = sum(abs(Y_pred-Y),'all');

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);


loss = double(loss);

end