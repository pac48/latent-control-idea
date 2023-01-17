function [loss, gradients, state] = simpleTFModelGradients(TFNet, Z, Tall)

loss = 0;
[map, state] = getNetOutput(TFNet, Z);

for tmp = Tall.keys
    object = tmp{1};
    key = [object '_TF_FC_Layer'];

    TPred = map(key);
    TPred = reshape(TPred, 6, 1, []);

    R = getR(TPred(1:3, :, :));
    P = TPred(4:6, :, :);
    out = Tall(object);
    inds = cellfun(@(x) ~isempty(x), out);

    R = R(:,:, inds);
    P = P(:,:, inds);
    T = cat(3, out{inds});

    if isempty(T)
        continue
    end

    loss = loss + sum( (R-T(1:3,1:3, :)).^2 +  (P-T(1:3,end, :)).^2, 'all');

end

loss = loss./size(Z, 2);

gradients = dlgradient(loss, TFNet.Learnables);
loss = double(loss);
