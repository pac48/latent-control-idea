function [val, gradients] = modelGradientsJac(net, X, ind)

% Forward data through network.
Y_pred = forward(net, X);

gradients = dlgradient(Y_pred(ind), net.Learnables);

val = Y_pred(ind);

end