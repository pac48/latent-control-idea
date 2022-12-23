function [loss,gradients, state] = modelGradients(dlnet, features, T)
    [X, state] = dlnet.forward(features);
    rpy = X(4:6,:);
    xyz = X(1:3,:);

    ct = cos(rpy);
    st = sin(rpy);
    R11 = ct(1, :).*ct(3, :).*ct(2, :) - st(1, :).*st(3, :);
    R12 = -ct(1, :).*ct(2, :).*st(3, :) - st(1, :).*ct(3, :);
    R13 = ct(1, :).*st(2, :);
    R21 = st(1, :).*ct(3, :).*ct(2, :) + ct(1, :).*st(3, :);
    R22 = -st(1, :).*ct(2, :).*st(3, :) + ct(1, :).*ct(3, :);
    R23 = st(1, :).*st(2, :);
    R31 = -st(2, :).*ct(3, :);
    R32 = st(2, :).*st(3, :);
    R33 = ct(2, :);
    ZERO = R33*0;
    ONE =  ZERO + 1;

    TPred = [R11; R21; R31; ZERO; R12; R22; R32; ZERO; R13; R23; R33; ZERO; xyz(1,:); xyz(2,:); xyz(3,:); ONE];


    loss = sum( (TPred - reshape(T, 16,[]) ).^2, 'all');
    gradients = dlgradient(loss, dlnet.Learnables);
    loss = double(loss);
end